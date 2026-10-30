"""``lucid.peft`` under ``lucid.compile``: the adapters train, and the forward follows the branch.

A LoRA layer is an ordinary ``nn.Module``, so the compiled entry points
take it as one — but its forward chooses a branch in Python
(``disable_adapters``, ``merged``) and the trace bakes the branch in.
Two properties are held here:

* **Training.**  ``make_step`` and ``fused_step`` over a model whose base
  is frozen train the adapters and nothing else, and match the eager
  loop step for step (AdamW, SGD; ``maximize`` honoured — LCD-687).
* **The forward.**  ``lucid.compile(model)`` equals eager with the
  adapter on, and a module compiled *before* ``disable_adapter()`` or
  ``merge_adapter()`` follows the switch: the layer keeps what it branches
  on in public scalar attributes, which the compile key reads (LCD-703).
  ``fused_step``'s plan key reads no module state at all, so its case is a
  strict xfail on LCD-704.  Compiling again after the toggle, or compiling
  ``merge_and_unload()``'s plain model, is asserted beside them.

Tolerance: ``make_step`` takes MPSGraph gradients into the eager
optimizer, ``fused_step`` takes the whole update from MPSGraph; both
round a few ulp differently from eager, so parameters agree to 1e-6
over a handful of steps, a forward to 1e-5.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft.tuners.lora import LoraLayer
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

TOL = 1e-6
FORWARD_TOL = 1e-5
STEPS = 5


pytestmark = pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")


class _Net(nn.Module):
    """Two linear layers around a tanh — the two LoRA targets."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).tanh())


def _peft(init_lora_weights: bool = True) -> PeftModel:
    """A seeded, adapted net on the compile device; the base is frozen."""
    lucid.manual_seed(0)
    config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["fc1", "fc2"], init_lora_weights=init_lora_weights
    )
    return get_peft_model(_Net(), config).to(COMPILE_DEVICE)


def _data() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return (lucid.randn(10, 6).to(COMPILE_DEVICE), lucid.randn(10, 3).to(COMPILE_DEVICE))


def _adapters(model: PeftModel) -> dict[str, lucid.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _base(model: PeftModel) -> dict[str, lucid.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}


def _drift(a: dict[str, lucid.Tensor], b: dict[str, lucid.Tensor]) -> float:
    assert a.keys() == b.keys()
    return max(float((a[k] - b[k]).abs().max().item()) for k in a)


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


class _Loop:
    """One adapted net trained eagerly, by ``make_step`` or by ``fused_step``."""

    def __init__(
        self,
        mode: str,
        make_opt: Callable[[list[lucid.Tensor]], optim.Optimizer],
        init_lora_weights: bool = True,
        every_parameter: bool = False,
    ) -> None:
        self.mode = mode
        self.model = _peft(init_lora_weights)
        self.model.train()
        params = [p for p in self.model.parameters() if every_parameter or p.requires_grad]
        self.opt = make_opt(params)
        self.fused = fused_step(self.model, F.mse_loss, self.opt) if mode == "fused" else None
        self.made = make_step(self.model, F.mse_loss) if mode == "made" else None

    def compiled(self) -> bool:
        """Whether the compiled step built an executable and fell back to nothing."""
        if self.fused is not None:
            return len(getattr(self.fused, "_plans")) == 1
        if self.made is not None:
            return not getattr(self.made, "eager_only")
        return True

    def step(self, x: lucid.Tensor, t: lucid.Tensor) -> None:
        if self.fused is not None:
            self.fused(x, t)
            return
        self.opt.zero_grad()
        out = self.made(x, t) if self.made is not None else F.mse_loss(self.model(x), t)
        out.backward()
        self.opt.step()


OPTIMIZERS = [
    pytest.param(lambda ps: optim.AdamW(ps, lr=0.05, weight_decay=0.1), id="AdamW"),
    pytest.param(lambda ps: optim.SGD(ps, lr=0.2, momentum=0.9), id="SGD-momentum"),
    pytest.param(
        lambda ps: optim.AdamW(ps, lr=0.02, weight_decay=0.1, maximize=True), id="AdamW-maximize"
    ),
    pytest.param(lambda ps: optim.SGD(ps, lr=0.05, maximize=True), id="SGD-maximize"),
]
COMPILED_MODES = ["made", "fused"]


@pytest.mark.parametrize("init_lora_weights", [True, False], ids=["B-zero", "B-random"])
@pytest.mark.parametrize("make_opt", OPTIMIZERS)
@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_a_compiled_step_trains_the_adapters_as_eager_does(
    mode: str, make_opt: object, init_lora_weights: bool
) -> None:
    """Adapter parameters follow the eager loop; the frozen base does not move."""
    x, t = _data()
    eager = _Loop("eager", make_opt, init_lora_weights)
    comp = _Loop(mode, make_opt, init_lora_weights)
    base_before = _base(comp.model)
    start = _adapters(comp.model)
    assert start, "no trainable parameter: the adapters were frozen"
    for k in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_adapters(eager.model), _adapters(comp.model))
        assert drift < TOL, f"step {k + 1}: adapter drift {drift:.3e}"
    assert _drift(base_before, _base(comp.model)) == 0.0, "a compiled step moved the frozen base"
    assert comp.compiled(), "the step ran eagerly, so the comparison proves nothing"
    assert _drift(start, _adapters(comp.model)) > 1e-3, "the adapters did not train"
    assert _drift(base_before, _base(eager.model)) == 0.0


@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_only_the_adapters_are_trainable_and_get_gradients(mode: str) -> None:
    """The base reaches the step as a frozen operand: no gradient lands on it."""
    x, t = _data()
    loop = _Loop(mode, lambda ps: optim.SGD(ps, lr=0.1), init_lora_weights=False)
    loop.step(x, t)
    for name, param in loop.model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad
            assert param.grad is None, f"{name} got a gradient"
    assert loop.compiled()
    if mode == "made":
        assert all(
            p.grad is not None for n, p in loop.model.named_parameters() if p.requires_grad
        ), "an adapter got no gradient"


@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_an_optimizer_over_every_parameter_leaves_the_frozen_base_alone(mode: str) -> None:
    """``AdamW(model.parameters())`` — frozen ones included — still trains the adapters only."""
    x, t = _data()

    def make(ps: list[lucid.Tensor]) -> optim.Optimizer:
        return optim.AdamW(ps, lr=0.05)

    eager = _Loop("eager", make, init_lora_weights=False, every_parameter=True)
    comp = _Loop(mode, make, init_lora_weights=False, every_parameter=True)
    base_before = _base(comp.model)
    for _ in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
    assert _drift(_adapters(eager.model), _adapters(comp.model)) < TOL
    assert _drift(base_before, _base(comp.model)) == 0.0


def test_a_fused_step_with_adapters_is_one_executable() -> None:
    x, t = _data()
    loop = _Loop("fused", lambda ps: optim.AdamW(ps, lr=0.05), init_lora_weights=False)
    for _ in range(3):
        loop.step(x, t)
    assert loop.compiled()


# ── the compiled forward ──────────────────────────────────────────


def _compiled_matches(
    compiled: Callable[[lucid.Tensor], lucid.Tensor], model: PeftModel, x: lucid.Tensor
) -> float:
    """Gap between ``compiled(x)`` and ``model(x)``, both under ``no_grad``."""
    with lucid.no_grad():
        return _gap(compiled(x), model(x))


def _trained(model: PeftModel) -> PeftModel:
    """``model`` after a few eager steps, so ``B`` is no longer zero."""
    x, t = _data()
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(3):
        opt.zero_grad()
        F.mse_loss(model(x), t).backward()
        opt.step()
    return model


def _without_adapter(model: PeftModel, x: lucid.Tensor) -> lucid.Tensor:
    with model.disable_adapter(), lucid.no_grad():
        return model(x)


def test_a_compiled_forward_applies_the_adapter() -> None:
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    compiled = lucid.compile(model)
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]
    with lucid.no_grad():
        # The adapter is in the compiled graph, not only in the eager one.
        assert _gap(compiled(x), _without_adapter(model, x)) > 1e-3


def test_a_compiled_module_called_under_grad_mode_trains_the_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under grad mode a model with trainable parameters runs eagerly (LCD-582), so ``backward`` works.

    The fallback is announced once per process and category, so the
    registry is emptied here: an earlier test that fell back for the same
    reason would otherwise use up the warning this one asserts.
    """
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    monkeypatch.setattr("lucid.compile._core.fallback._NOTICED", set())
    model = _peft(False).train()
    twin = _peft(False).train()
    x, t = _data()
    compiled = lucid.compile(model)
    with pytest.warns(CompileFallbackWarning, match="needs gradients"):
        out = compiled(x)
    F.mse_loss(out, t).backward()
    F.mse_loss(twin(x), t).backward()
    mine = dict(model.named_parameters())
    for name, param in twin.named_parameters():
        if param.requires_grad:
            assert mine[name].grad is not None
            assert _gap(mine[name].grad, param.grad) < 1e-6
        else:
            assert mine[name].grad is None


def test_a_loaded_adapter_compiles_under_grad_mode(tmp_path: object) -> None:
    """``from_pretrained`` freezes the adapter, so nothing requires grad and the graph is used."""
    trained = _trained(_peft(True)).eval()
    trained.save_pretrained(str(tmp_path))
    lucid.manual_seed(0)  # the seed ``_peft`` builds its base with
    loaded = PeftModel.from_pretrained(_Net().to(COMPILE_DEVICE), str(tmp_path))
    x, _ = _data()
    compiled = lucid.compile(loaded)
    got = compiled(x)
    assert _gap(got, loaded(x)) < FORWARD_TOL
    assert _gap(got, trained(x)) < FORWARD_TOL
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]


def test_merge_and_unload_compiles_as_a_plain_model() -> None:
    """The deployment path: fold the adapter in, then compile an ordinary model."""
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    with lucid.no_grad():
        adapted = model(x)
    plain = model.merge_and_unload()
    assert not any(isinstance(m, LoraLayer) for m in plain.modules())
    plain.eval()
    compiled = lucid.compile(plain)
    with lucid.no_grad():
        got = compiled(x)
    assert _gap(got, adapted) < FORWARD_TOL
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]


# ── a toggle after the module was compiled (LCD-703 / LCD-704) ─────


def _warmed() -> tuple[PeftModel, Callable[[lucid.Tensor], lucid.Tensor], lucid.Tensor]:
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    compiled = lucid.compile(model)
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL
    return model, compiled, x


def test_disable_adapter_reaches_a_module_compiled_before_it() -> None:
    model, compiled, x = _warmed()
    with model.disable_adapter(), lucid.no_grad():
        want = model(x)
        got = compiled(x)
    assert _gap(got, want) < FORWARD_TOL


def test_merge_adapter_reaches_a_module_compiled_before_it() -> None:
    model, compiled, x = _warmed()
    model.base_model.merge_adapter()
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL


def test_the_adapter_coming_back_is_followed_after_a_disabled_call() -> None:
    """Compiled first while disabled: the enabled model must not get the disabled graph."""
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    with model.disable_adapter():
        compiled = lucid.compile(model)
        assert _compiled_matches(compiled, model, x) < FORWARD_TOL
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL


def test_unmerge_adapter_reaches_a_module_compiled_while_merged() -> None:
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    model.base_model.merge_adapter()
    compiled = lucid.compile(model)
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL
    model.base_model.unmerge_adapter()
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL


@pytest.mark.parametrize(
    "mode",
    [
        "made",
        pytest.param(
            "fused",
            marks=pytest.mark.xfail(
                strict=True,
                reason="LCD-704: fused_step's plan key holds no module state (training "
                "flags, public scalar attributes), so a step traced with the adapter on is "
                "replayed after disable_adapter()",
            ),
        ),
    ],
)
def test_disable_adapter_reaches_a_compiled_training_step(mode: str) -> None:
    """The same switch in ``make_step`` / ``fused_step``: the loss must drop the adapter."""
    x, t = _data()
    loop = _Loop(mode, lambda ps: optim.SGD(ps, lr=0.0), init_lora_weights=False)
    loop.step(x, t)  # traced with the adapter on
    with loop.model.disable_adapter():
        want = float(F.mse_loss(loop.model(x), t).item())
        step = loop.made if loop.made is not None else loop.fused
        assert step is not None
        got = float(step(x, t).item())
    assert abs(got - want) < 1e-5


def test_compiling_again_after_disable_adapter_is_right() -> None:
    """The safe path today: a ``lucid.compile`` made inside the block traces the base alone."""
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    with lucid.no_grad():
        adapted = model(x)
    with model.disable_adapter():
        compiled = lucid.compile(model)
        assert _compiled_matches(compiled, model, x) < FORWARD_TOL
        with lucid.no_grad():
            assert _gap(compiled(x), adapted) > 1e-3


def test_compiling_again_after_merge_adapter_is_right() -> None:
    model = _trained(_peft(True)).eval()
    x, _ = _data()
    with lucid.no_grad():
        adapted = model(x)
    model.base_model.merge_adapter()
    compiled = lucid.compile(model)
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL
    # Merged and enabled are the same function.
    with lucid.no_grad():
        assert _gap(compiled(x), adapted) < FORWARD_TOL


def test_a_toggle_is_undone_by_going_back_to_the_traced_state() -> None:
    """Back in the state it was compiled in, the cached graph is the right one again."""
    model, compiled, x = _warmed()
    model.base_model.merge_adapter()
    model.base_model.unmerge_adapter()
    assert _compiled_matches(compiled, model, x) < FORWARD_TOL
