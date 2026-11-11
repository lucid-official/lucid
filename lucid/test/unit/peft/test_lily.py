"""Lily: the routed update in closed form, which layers share what, the refusal to merge, the file and the reference's numbers.

Lily (Zhong et al., 2024) projects down with an ``A`` shared by a run of layers, mixes up-projections ``B_e`` shared by all
layers with the softmax of a per-layer router (averaged over every row of the batch, Fig. 7) and adds ``s * h * sum_e S_e B_e``.
Because the mixture depends on the batch, no weight holds the update: a Lily layer cannot merge.  The tests expect the update
from plain loops (``_bank_forms``) and the sharing from the identity of the modules; the shared contract
(``test_tuner_contract``) runs here under Lily's own ``method`` fixture, without its merging tests.  The ``parity`` tests build
the same adapter in the reference library and compare.
"""

from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    LilyConfig,
    LilyModel,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners import lily
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.peft import _bank_forms as forms
from lucid.test.unit.peft._closed_forms import _PREFIX, CLOSED_FORMS, _mlp
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft.test_low_rank_delta_contract import TestForward
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
    TestMultiAdapter,
    TestRegistration,
    _Method,
    test_a_reference_written_adapter_loads,
)

#: The shared contract's tests, collected here under this module's ``method`` fixture.  ``TestMerging`` is not among
#: them: a Lily layer cannot merge (``TestRefusesToMerge`` holds that).
__all__ = [
    "TestFiles",
    "TestForward",
    "TestMultiAdapter",
    "TestRegistration",
    "test_a_reference_written_adapter_loads",
]


def _config(**kw: object) -> LilyConfig:
    fields: dict[str, object] = {"r": 3, "num_B": 2, "target_modules": ["0", "2"]} | kw
    return LilyConfig.from_dict(fields)


def _lily(targets: tuple[str, ...], random: bool, **kw: object) -> LilyConfig:
    return _config(scaling=2.0, target_modules=list(targets), init_weights=not random, **kw)


def _lily_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Rank 3, two up-projections, scale 2: each layer adds ``2 h sum_e S_e B_e``, ``S`` the router's softmax averaged over the rows."""
    net = _mlp()
    hidden = x
    for index in (0, 2):
        where = f"{_PREFIX}{index}."
        update = forms.lily_update(
            hidden.tolist(),
            t[where + "lily_A.weight"].tolist(),
            t[where + "lily_B.weight"].tolist(),
            t[where + "lily_router.weight"].tolist(),
            2,
            2.0,
        )
        hidden = hidden @ net[index].weight.mT + net[index].bias + lucid.tensor(update)
        if index == 0:
            hidden = nn.functional.relu(hidden)
    return hidden


_FILE = {
    f"{_PREFIX}0.lily_A.weight": (3, 6),
    f"{_PREFIX}0.lily_B.weight": (6, 8),
    f"{_PREFIX}0.lily_router.weight": (2, 3),
    f"{_PREFIX}2.lily_A.weight": (3, 8),
    f"{_PREFIX}2.lily_B.weight": (6, 4),
    f"{_PREFIX}2.lily_router.weight": (2, 3),
}
_METHODS = [_Method("lily", PeftType.LILY, _lily, LilyModel, lily.LilyLayer, _FILE)]

CLOSED_FORMS["lily"] = _lily_forward
_REFERENCE_CONFIGS["lily"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "exclude_modules": None,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "num_B": 2,
    "peft_type": "LILY",
    "peft_version": "0.21.2",
    "r": 3,
    "revision": None,
    "scaling": 2.0,
    "stride_A": 1,
    "target_modules": ["0", "2"],
    "task_type": None,
}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _randomise(model: nn.Module, seed: int = 3) -> None:
    """Every trainable parameter well away from its start, the up-projections included."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(lucid.randn(*p.shape, device=p.device))


def _stack(depth: int = 4, width: int = 6) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(*[nn.Linear(width, width) for _ in range(depth)])


class _Blocks(nn.Module):
    """``depth`` blocks, each holding a ``q`` projection: the layers one ``target_modules`` entry (``"q"``) selects, which Lily groups."""

    def __init__(self, depth: int, width: int = 6) -> None:
        super().__init__()
        lucid.manual_seed(0)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"q": nn.Linear(width, width)}) for _ in range(depth)]
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        for block in self.blocks:
            x = block["q"](x)
        return x


def _layer(peft: PeftModel, index: str) -> lily.LilyLayer:
    layer = peft.base_model.model.get_submodule(index)
    assert isinstance(layer, lily.LilyLayer)
    return layer


def _q(peft: PeftModel, index: int) -> lily.LilyLayer:
    return _layer(peft, f"blocks.{index}.q")


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = LilyConfig(target_modules=["a"])
        assert (config.r, config.stride_A, config.num_B, config.scaling) == (32, 1, 2, 1.0)
        assert config.init_weights is True and config.exclude_modules is None

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            LilyConfig(num_B=1, target_modules=["a"])
        with pytest.raises(ValueError, match="positive"):
            LilyConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="positive"):
            LilyConfig(stride_A=0, target_modules=["a"])


class TestUpdate:
    def _layer(self) -> tuple[nn.Linear, lily.Linear]:
        lucid.manual_seed(0)
        base = nn.Linear(8, 4)
        layer = lily.Linear(
            base, "default", _config(scaling=1.5, init_weights=False, target_modules=["x"])
        )
        return base, layer

    def test_the_update_is_the_scaled_hidden_through_the_routed_up_projections(self) -> None:
        base, layer = self._layer()
        x = lucid.randn(5, 8)
        expected = forms.lily_update(
            x.tolist(),
            layer.lily_A["default"].weight.tolist(),
            layer.lily_B["default"].weight.tolist(),
            layer.lily_router["default"].weight.tolist(),
            2,
            1.5,
        )
        assert _span(layer(x), base(x) + lucid.tensor(expected)) < 1e-5

    def test_the_mixture_is_the_router_average_over_the_whole_batch(self) -> None:
        """Fig. 7: ``S`` is one vector for the call, so a row's output depends on the other rows."""
        _, layer = self._layer()
        x = lucid.randn(5, 8)
        assert _span(layer(x)[:1], layer(x[:1])) > 1e-3
        # ... and sequence positions count as rows of the same batch.
        sequence = lucid.randn(2, 3, 8)
        flat = layer(sequence.reshape(6, 8)).reshape(2, 3, 4)
        assert _span(layer(sequence), flat) < 1e-5

    def test_the_default_start_is_the_base_bit_for_bit(self) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 4)
        layer = lily.Linear(base, "default", _config(target_modules=["x"]))
        x = lucid.randn(5, 8)
        assert _same(layer(x), base(x))

    def test_the_projections_are_bias_free_linear_layers_of_the_references_shapes(self) -> None:
        layer = lily.Linear(nn.Linear(8, 4), "default", _config(r=3, num_B=4, target_modules=["x"]))
        down, up, router = (
            layer.lily_A["default"],
            layer.lily_B["default"],
            layer.lily_router["default"],
        )
        assert (tuple(down.weight.shape), tuple(up.weight.shape)) == ((3, 8), (12, 4))
        assert tuple(router.weight.shape) == (4, 3)
        assert down.bias is None and up.bias is None and router.bias is None

    def test_two_active_adapters_add(self) -> None:
        peft = get_peft_model(
            _stack(2), _config(init_weights=False, target_modules=["0", "1"]), adapter_name="a"
        )
        peft.add_adapter("b", _config(init_weights=False, target_modules=["0", "1"]))
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            only_a = peft(x)
            peft.set_adapter(["a", "b"])
            both = peft(x)
            assert not _same(both, only_a)
            peft.set_adapter("b")
            assert not _same(peft(x), only_a)


class TestSharing:
    """The tuner's grouping: by the ``target_modules`` entry and the weight's shape, in the model's order."""

    def test_a_down_projection_is_shared_by_a_run_of_stride_a_layers(self) -> None:
        peft = get_peft_model(_Blocks(5), _config(stride_A=2, target_modules=["q"]))
        down = [_q(peft, i).lily_A["default"] for i in range(5)]
        assert down[0] is down[1] and down[2] is down[3]
        assert len({id(m) for m in down}) == 3  # (0, 1), (2, 3), (4)

    def test_the_up_projections_are_shared_by_every_layer_of_a_group(self) -> None:
        peft = get_peft_model(_Blocks(4), _config(target_modules=["q"]))
        up = {id(_q(peft, i).lily_B["default"]) for i in range(4)}
        routers = {id(_q(peft, i).lily_router["default"]) for i in range(4)}
        assert len(up) == 1 and len(routers) == 4  # a router is each layer's own

    def test_entries_that_name_each_layer_alone_share_nothing(self) -> None:
        """The group is the entry that selected the layer: ``"0"`` and ``"1"`` are two groups."""
        peft = get_peft_model(_stack(2), _config(target_modules=["0", "1"]))
        assert _layer(peft, "0").lily_B["default"] is not _layer(peft, "1").lily_B["default"]

    def test_layers_that_a_different_entry_selects_do_not_share(self) -> None:
        lucid.manual_seed(0)
        blocks = nn.ModuleList(
            [nn.ModuleDict({"q": nn.Linear(6, 6), "v": nn.Linear(6, 6)}) for _ in range(3)]
        )
        peft = get_peft_model(blocks, _config(target_modules=["q", "v"]))
        q = [peft.base_model.model[i]["q"].lily_B["default"] for i in range(3)]
        v = [peft.base_model.model[i]["v"].lily_B["default"] for i in range(3)]
        assert q[0] is q[1] is q[2] and v[0] is v[1] is v[2] and q[0] is not v[0]

    def test_layers_of_another_shape_do_not_share(self) -> None:
        lucid.manual_seed(0)
        blocks = nn.ModuleList([nn.ModuleDict({"q": nn.Linear(n, n)}) for n in (6, 6, 8, 8)])
        peft = get_peft_model(blocks, _config(target_modules=["q"]))
        up = [peft.base_model.model[i]["q"].lily_B["default"] for i in range(4)]
        assert up[0] is up[1] and up[2] is up[3] and up[0] is not up[2]

    def test_a_shared_module_trains_once_and_every_layer_feeds_it(self) -> None:
        peft = get_peft_model(
            _Blocks(4), _config(stride_A=2, init_weights=False, target_modules=["q"])
        )
        trainable = [p for p in peft.parameters() if p.requires_grad]
        # down (2 runs) + up (1) + routers (4)
        assert len(trainable) == 2 + 1 + 4
        peft(lucid.randn(5, 6)).sum().backward()
        assert all(p.grad is not None and float(p.grad.abs().max().item()) > 0 for p in trainable)

    def test_an_adapter_added_later_is_shared_on_its_own_layers(self) -> None:
        first = _config(layers_to_transform=[0, 1], layers_pattern="blocks", target_modules=["q"])
        later = _config(
            layers_to_transform=[1, 2, 3], layers_pattern="blocks", target_modules=["q"]
        )
        peft = get_peft_model(_Blocks(4), first, adapter_name="a")
        peft.add_adapter("b", later)
        up_a = [_q(peft, i).lily_B["a"] for i in (0, 1)]
        up_b = [_q(peft, i).lily_B["b"] for i in (1, 2, 3)]
        assert up_a[0] is up_a[1] and up_b[0] is up_b[1] is up_b[2] and up_a[0] is not up_b[0]


class TestRefusesToMerge:
    def test_merging_says_why(self) -> None:
        peft = get_peft_model(_mlp(), _config(init_weights=False))
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            peft.base_model.merge_adapter()
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            peft.merge_and_unload()

    def test_the_layer_has_no_weight_form(self) -> None:
        layer = lily.Linear(nn.Linear(8, 4), "default", _config(target_modules=["x"]))
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            layer.get_delta_weight("default")

    def test_a_refused_merge_changes_nothing_and_unload_still_works(self) -> None:
        net = _mlp()
        weights = [p.detach().clone() for p in net.parameters()]
        peft = get_peft_model(net, _config(init_weights=False))
        own = [
            p
            for _, layer in peft.named_modules()
            if isinstance(layer, lily.LilyLayer)
            for p in layer.get_base_layer().parameters()
        ]
        with pytest.raises(NotImplementedError):
            peft.base_model.merge_adapter()
        assert all(_same(a, b) for a, b in zip(own, weights, strict=True))
        assert peft.unload() is net and type(net[0]) is nn.Linear


class TestCompile:
    @pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")
    def test_compiled_follows_set_adapter_and_disable(self) -> None:
        net = _mlp().to("metal")
        peft = get_peft_model(net, _config(init_weights=False), adapter_name="a")
        peft.add_adapter("b", _config(init_weights=False, target_modules=["0"]))
        _randomise(peft)
        compiled = lucid.compile(peft)
        x = lucid.randn(4, 6, device="metal")
        with lucid.no_grad():
            on_a = peft(x)
            assert _span(compiled(x), on_a) < 1e-4
            peft.set_adapter("b")
            on_b = peft(x)
            assert _span(on_a, on_b) > 1e-3 and _span(compiled(x), on_b) < 1e-4
            with peft.disable_adapter():
                assert _span(compiled(x), peft(x)) < 1e-4


class TestFilesOfSharedModules:
    def test_every_layer_writes_the_modules_it_reads(self) -> None:
        peft = get_peft_model(_Blocks(2), _config(stride_A=2, target_modules=["q"]))
        state = get_peft_model_state_dict(peft)
        assert sorted(state) == [
            f"base_model.model.blocks.{i}.q.lily_{name}.weight"
            for i in "01"
            for name in ("A", "B", "router")
        ]
        down = "base_model.model.blocks.{}.q.lily_A.weight"
        assert _same(state[down.format(0)], state[down.format(1)])

    def test_loading_restores_the_shared_modules(self, tmp_path: Path) -> None:
        trained = get_peft_model(_Blocks(4), _config(stride_A=2, target_modules=["q"]))
        _randomise(trained)
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_Blocks(4), tmp_path)
        with lucid.no_grad():
            assert _same(loaded(x), expected)
        assert _q(loaded, 0).lily_A["default"] is _q(loaded, 1).lily_A["default"]


def _identities(layers: list[nn.Module], attribute: str) -> list[int]:
    """The pattern of sharing among ``layers``: for each, the position of the first layer whose module it shares."""
    held = [getattr(layer, attribute)["default"] for layer in layers]
    return [next(j for j in range(len(held)) if held[j] is held[i]) for i in range(len(held))]


@pytest.mark.parity
class TestAgainstTheReferenceLibrary:
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_the_same_layers_share_the_same_modules(self, ref: ModuleType, stride: int) -> None:
        """Seven blocks of ``q`` and ``v`` projections, two shapes: the reference's grouping, layer for layer."""
        library = adapter_library()
        widths = (6, 6, 6, 8, 8, 6, 6)
        theirs = library.get_peft_model(
            ref.nn.ModuleList(
                [
                    ref.nn.ModuleDict({"q": ref.nn.Linear(n, n), "v": ref.nn.Linear(6, 6)})
                    for n in widths
                ]
            ),
            library.LilyConfig(r=2, stride_A=stride, target_modules=["q", "v"]),
        )
        lucid.manual_seed(0)
        mine = get_peft_model(
            nn.ModuleList(
                [nn.ModuleDict({"q": nn.Linear(n, n), "v": nn.Linear(6, 6)}) for n in widths]
            ),
            _config(r=2, stride_A=stride, target_modules=["q", "v"]),
        )
        for kind in ("q", "v"):
            ours = [mine.base_model.model[i][kind] for i in range(len(widths))]
            theirs_layers = [theirs.base_model.model[i][kind] for i in range(len(widths))]
            for attribute in ("lily_A", "lily_B"):
                assert _identities(ours, attribute) == _identities(theirs_layers, attribute)

    def test_an_adapter_the_reference_wrote_loads_and_computes_the_same(
        self, ref: ModuleType, tmp_path: Path
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
            library.LilyConfig(
                r=3, stride_A=1, num_B=2, scaling=2.0, target_modules=["0", "2"], init_weights=False
            ),
        )
        theirs.save_pretrained(tmp_path)
        x = ref.randn(5, 6)
        with ref.no_grad():
            want = theirs(x).tolist()
        loaded = PeftModel.from_pretrained(net, tmp_path)
        with lucid.no_grad():
            assert _span(loaded(lucid.tensor(x.tolist())), lucid.tensor(want)) < 1e-5
