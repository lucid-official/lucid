"""A compiled call reads the tensors and modules the model holds now (LCD-560).

An executable is fed the parameters and buffers it was traced with and runs
the forwards of the modules that were in the tree then.  A tensor or a
submodule put in another's place leaves the cache key as it was — same
shapes, dtypes and devices — so on Metal the compiled call kept reading the
old ones until a new input shape forced a trace:

    m.s = nn.Parameter(lucid.full((4,), 5.0))   # compiled: 3.0, eager: 5.0

``register_parameter`` / ``register_buffer`` did not even move the tree
version.  Each test below makes one replacement after the first compiled
call — of a parameter, a buffer or a submodule, by every route there is —
and holds the compiled answer to eager, on the CPU (which runs eagerly) and
on Metal.  The steps are held to the eager training loop: the gradient, and
the update, go to the parameter the model holds now.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import CompiledModule, compile_optimizer, fused_step, make_step
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test._fixtures.devices import metal_available
from lucid.utils.cache import StaticCache

pytestmark = pytest.mark.filterwarnings(
    "ignore::lucid.compile._core.fallback.CompileFallbackWarning"
)

_TOL = 1e-5
_METAL = pytest.mark.skipif(not metal_available(), reason="metal unavailable")
_DEVICES = ["cpu", pytest.param("metal", marks=_METAL)]


def _values(shape: tuple[int, ...], offset: float, device: str) -> lucid.Tensor:
    """A deterministic tensor, different for each ``offset``."""
    n = int(np.prod(shape))
    data = (np.arange(n, dtype=np.float32).reshape(shape) * 0.05 + offset) % 1.3
    return lucid.tensor(data).to(device)


def _linear(seed: int, device: str) -> nn.Linear:
    lucid.manual_seed(seed)
    return nn.Linear(4, 4).to(device)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(4, 4)
        self.act = nn.Tanh()
        self.out = nn.Linear(4, 4)
        self.scales = nn.ParameterList([nn.Parameter(lucid.full((4,), 1.5))])
        self.register_buffer("shift", lucid.full((4,), 0.5))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.act(self.inp(x)) * self.scales[0]
        return self.out(h) + self.shift


def _net(device: str) -> _Net:
    lucid.manual_seed(0)
    return _Net().to(device)


def _data(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(3)
    x = lucid.tensor(rng.standard_normal((8, 4)).astype(np.float32)).to(device)
    t = lucid.tensor(rng.standard_normal((8, 4)).astype(np.float32)).to(device)
    return x, t


def _assign_load(m: _Net, device: str) -> None:
    m.inp.load_state_dict(
        {
            "weight": _values((4, 4), 0.7, device),
            "bias": _values((4,), 0.2, device),
        },
        assign=True,
    )


# What happens to the model after the first compiled call: (class, route).
_Change = Callable[[_Net, str], None]
_CHANGES: dict[str, _Change] = {
    "parameter-setattr": lambda m, d: setattr(
        m.inp, "weight", nn.Parameter(_values((4, 4), 0.3, d))
    ),
    "parameter-register_parameter": lambda m, d: m.out.register_parameter(
        "bias", nn.Parameter(_values((4,), 0.9, d))
    ),
    "parameter-ParameterList-item": lambda m, d: m.scales.__setitem__(
        0, nn.Parameter(_values((4,), 0.4, d))
    ),
    "parameter-weight-tying": lambda m, d: setattr(m.out, "weight", m.inp.weight),
    "parameter-assign-load": _assign_load,
    "buffer-setattr": lambda m, d: setattr(m, "shift", _values((4,), -0.6, d)),
    "buffer-register_buffer": lambda m, d: m.register_buffer(
        "shift", _values((4,), 0.8, d)
    ),
    "submodule-with-parameters": lambda m, d: setattr(m, "inp", _linear(7, d)),
    # Tanh and Sigmoid key alike: neither has a flag or a scalar attribute.
    "submodule-without-parameters": lambda m, d: setattr(m, "act", nn.Sigmoid()),
}


def _close(a: lucid.Tensor, b: lucid.Tensor) -> None:
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=_TOL, atol=_TOL)


def _reload(model: nn.Module, device: str) -> None:
    del device
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, assign=True)


def _round_trip(model: nn.Module, device: str) -> None:
    # A ``.to()`` to where the model is replaces nothing and tells no one
    # (LCD-601), so the notice comes from a cast there and back.
    del device
    model.half()
    model.float()


# When the replacement comes: before which call, and right after which notice.
# Before the first call matters — ``make_step`` takes the parameters, and the
# segmented step its children, when it is built — and so does right after an
# assign load or a conversion: told of those, the owner takes the tree again.
_WHEN: dict[str, tuple[int, Callable[[nn.Module, str], None] | None]] = {
    "before-the-first-call": (0, None),
    "after-the-first-call": (2, None),
    "right-after-an-assign-load": (0, _reload),
    "right-after-a-dtype-round-trip": (2, _round_trip),
}


def _replace[M: nn.Module](
    model: M,
    changes: dict[str, Callable[[M, str], None]],
    change: str,
    when: str,
    device: str,
) -> None:
    notice = _WHEN[when][1]
    if notice is not None:
        notice(model, device)
    changes[change](model, device)


# A module the compiled function reaches through a module-level name, not a
# closure cell or a bound ``self`` (LCD-599).
_GLOBAL_NET: nn.Module = nn.Identity()


def _global_net_call(x: lucid.Tensor) -> lucid.Tensor:
    return _GLOBAL_NET(x)


def _compiled(
    model: nn.Module, entry: str
) -> CompiledModule[[lucid.Tensor], lucid.Tensor]:
    """``lucid.compile`` of the model, of its bound forward, of a closure, or
    of a function that reads it as a global."""
    global _GLOBAL_NET
    if entry == "module":
        return lucid.compile(model)
    if entry == "bound-method":
        return lucid.compile(model.forward)
    if entry == "global":
        _GLOBAL_NET = model
        return lucid.compile(_global_net_call)

    def call(x: lucid.Tensor) -> lucid.Tensor:
        return model(x)

    return lucid.compile(call)


_ENTRIES = ["module", "bound-method", "closure", "global"]


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("when", list(_WHEN))
@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("change", list(_CHANGES))
def test_a_compiled_module_answers_with_what_the_model_holds_now(
    change: str, entry: str, when: str, device: str
) -> None:
    model = _net(device)
    x, _ = _data(device)
    compiled = _compiled(model, entry)
    for _ in range(_WHEN[when][0]):
        _close(compiled(x), model(x))
    _replace(model, _CHANGES, change, when, device)
    _close(compiled(x), model(x))
    if device == "metal":
        # It answered compiled, traced again — not eagerly.
        info = compiled.cache_info()
        assert info["entries"] == 1 and not info["eager_only"], info


def _named(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in model.named_parameters()}


def _assert_compiled(cache: object, eager_only: object) -> None:
    """A step that answered eagerly would match eager without being fixed."""
    assert cache and not eager_only, (cache, eager_only)


def _make_step_run(
    model: _Net, change: str, when: str, device: str, compiled: bool
) -> list[float]:
    x, t = _data(device)
    step = make_step(model, F.mse_loss) if compiled else None
    opt = optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for k in range(4):
        if k == _WHEN[when][0]:
            _replace(model, _CHANGES, change, when, device)
            # A training loop steps what the model holds after a replacement.
            opt = optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss = step(x, t) if step is not None else F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    if step is not None and device == "metal":
        _assert_compiled(step.cache, len(step.eager_only))  # type: ignore[attr-defined]
    return losses


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("when", list(_WHEN))
@pytest.mark.parametrize("change", list(_CHANGES))
def test_make_step_trains_what_the_model_holds_now(
    change: str, when: str, device: str
) -> None:
    model, ref = _net(device), _net(device)
    losses = _make_step_run(model, change, when, device, compiled=True)
    ref_losses = _make_step_run(ref, change, when, device, compiled=False)
    np.testing.assert_allclose(losses, ref_losses, rtol=_TOL, atol=_TOL)
    got, want = _named(model), _named(ref)
    assert got.keys() == want.keys()
    for name, value in want.items():
        np.testing.assert_allclose(got[name], value, rtol=_TOL, atol=_TOL, err_msg=name)


@_METAL
@pytest.mark.parametrize("when", list(_WHEN))
@pytest.mark.parametrize("change", list(_CHANGES))
def test_fused_step_trains_what_the_model_holds_now(change: str, when: str) -> None:
    # fused_step steps the optimizer it was built with; after a replacement
    # the loop hands that optimizer the model's parameters, and the fused
    # step is held to the eager loop doing the same.  (fused_step has no
    # CPU path: the CPU is refused before any of this.)
    device = "metal"
    x, t = _data(device)
    model, ref = _net(device), _net(device)
    opt = optim.SGD(model.parameters(), lr=0.1)
    step = fused_step(model, F.mse_loss, opt)
    ref_opt = optim.SGD(ref.parameters(), lr=0.1)
    for k in range(4):
        if k == _WHEN[when][0]:
            for m, o in ((model, opt), (ref, ref_opt)):
                _replace(m, _CHANGES, change, when, device)
                o.param_groups[0]["params"] = list(m.parameters())
        loss = float(step(x, t).item())
        ref_opt.zero_grad()
        ref_loss = F.mse_loss(ref(x), t)
        ref_loss.backward()
        ref_opt.step()
        np.testing.assert_allclose(loss, float(ref_loss.item()), rtol=_TOL, atol=_TOL)
    info = step.cache_info()
    _assert_compiled(info["entries"], info["eager_only"])
    got, want = _named(model), _named(ref)
    for name, value in want.items():
        np.testing.assert_allclose(got[name], value, rtol=_TOL, atol=_TOL, err_msg=name)


@_METAL
def test_a_compiled_optimizer_steps_a_parameter_put_in_anothers_place() -> None:
    # The groups keep their sizes, so a check by size saw nothing and the
    # step went on updating the parameter that left — with its stale grad.
    device = "metal"
    runs = []
    for compiled in (True, False):
        old = nn.Parameter(lucid.full((3,), 1.0, device=device))
        opt = optim.Adam([old], lr=0.1)
        stepper = compile_optimizer(opt) if compiled else opt
        (old * old).sum().backward()
        stepper.step()
        new = nn.Parameter(lucid.full((3,), 4.0, device=device))
        opt.param_groups[0]["params"][0] = new
        for _ in range(2):
            new.grad = None
            (new * new).sum().backward()
            stepper.step()
        runs.append((old.detach().numpy(), new.detach().numpy()))
    (old_c, new_c), (old_e, new_e) = runs
    np.testing.assert_allclose(old_c, old_e, rtol=_TOL, atol=_TOL)
    np.testing.assert_allclose(new_c, new_e, rtol=_TOL, atol=_TOL)


def _seq(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4)).to(device)


_SEQ_CHANGES: dict[str, Callable[[nn.Sequential, str], None]] = {
    "parameter-setattr": lambda m, d: setattr(
        m[0], "weight", nn.Parameter(_values((4, 4), 0.3, d))
    ),
    "submodule-with-parameters": lambda m, d: m.__setitem__(2, _linear(7, d)),
    "submodule-without-parameters": lambda m, d: m.__setitem__(1, nn.Sigmoid()),
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("when", list(_WHEN))
@pytest.mark.parametrize("change", list(_SEQ_CHANGES))
def test_a_segmented_step_trains_what_the_model_holds_now(
    change: str, when: str, device: str
) -> None:
    # Each segment held the children and parameters it was split from.
    x, t = _data(device)
    model, ref = _seq(device), _seq(device)
    step = make_step(model, F.mse_loss, segments=2)
    for k in range(4):
        if k == _WHEN[when][0]:
            _replace(model, _SEQ_CHANGES, change, when, device)
            _replace(ref, _SEQ_CHANGES, change, when, device)
        for p in (*model.parameters(), *ref.parameters()):
            p.grad = None
        loss = step(x, t)
        loss.backward()
        ref_loss = F.mse_loss(ref(x), t)
        ref_loss.backward()
        np.testing.assert_allclose(
            float(loss.item()), float(ref_loss.item()), rtol=_TOL, atol=_TOL
        )
        for p, rp in zip(model.parameters(), ref.parameters(), strict=True):
            assert p.grad is not None and rp.grad is not None
            np.testing.assert_allclose(
                p.grad.numpy(), rp.grad.numpy(), rtol=_TOL, atol=_TOL
            )
    if device == "metal":
        segment_caches = step.cache  # type: ignore[attr-defined]
        eager = step.eager_only["flag"]  # type: ignore[attr-defined]
        _assert_compiled(all(segment_caches), eager)


class _Decoder(nn.Module):
    """A single-head attention step over a StaticCache, as a decoder runs."""

    def __init__(self) -> None:
        super().__init__()
        self.kv = nn.Linear(4, 4)
        self.o = nn.Linear(4, 4)

    def forward(
        self, hidden: lucid.Tensor, cache: StaticCache, position: lucid.Tensor
    ) -> lucid.Tensor:
        kv = self.kv(hidden).reshape(1, 1, 1, 4)
        keys, values = cache.update(kv, kv, 0, {"cache_position": position})
        scores = hidden.reshape(1, 1, 1, 4) @ keys.permute(0, 1, 3, 2)
        idx = lucid.arange(_SLOTS, device=hidden.device.type).long()
        valid = idx.reshape(1, 1, 1, _SLOTS) <= position.reshape(1, 1, 1, 1)
        scores = lucid.where(valid, scores, lucid.tensor(-1e30, device="metal"))
        return self.o((F.softmax(scores, dim=-1) @ values).reshape(1, 1, 4))


_SLOTS = 6


def _decode_fn(
    model: _Decoder, cache: StaticCache
) -> Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]:
    def fwd(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
        return model(h, cache, cp)

    return fwd


@pytest.mark.usefixtures("inference")
@_METAL
def test_a_compiled_decode_step_reads_a_replaced_parameter() -> None:
    outs: list[list[np.ndarray]] = []
    for compiled in (True, False):
        lucid.manual_seed(0)
        model = _Decoder().to("metal").eval()
        cache = StaticCache(_SLOTS)
        fwd = _decode_fn(model, cache)
        step = compiled_decode_step(fwd, cache) if compiled else fwd
        run = []
        for pos in range(_SLOTS):
            if pos == 3:
                model.o.weight = nn.Parameter(_values((4, 4), 0.6, "metal"))
            h = _values((1, 1, 4), 0.1 * (pos + 1), "metal")
            cp = lucid.tensor([pos], device="metal").long()
            run.append(step(h, cp).numpy())
        if compiled:
            assert len(step.cache) == 1  # type: ignore[attr-defined]
        outs.append(run)
    for got, want in zip(*outs, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


# ── What a wrapped function reaches, and plain tensor attributes ───────────


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.parametrize("entry", ["module", "bound-method", "closure", "global"])
def test_the_training_flag_of_a_module_reached_any_way_is_keyed(entry: str) -> None:
    # Dropout is on until the change turns it off.
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(0.5)).to("metal")
    x, _ = _data("metal")
    compiled = _compiled(model, entry)
    compiled(x)
    model.eval()
    _close(compiled(x), model(x))
    assert compiled.cache_info()["entries"] == 2


class _Masked(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        # A plain attribute, not a buffer: it lives in ``__dict__``.
        self.mask = lucid.ones(4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.lin(x) * self.mask


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_plain_tensor_attribute_put_in_anothers_place(entry: str) -> None:
    # The trace was fed it from outside, so the tree record holds its slot
    # (LCD-606); setting another moves the tree version, as any setattr does.
    lucid.manual_seed(0)
    model = _Masked().to("metal")
    model.mask = model.mask.to("metal")
    x, _ = _data("metal")
    compiled = _compiled(model, entry)
    compiled(x)
    model.mask = _values((4,), 0.9, "metal")
    _close(compiled(x), model(x))
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


@pytest.mark.parametrize("device", _DEVICES)
def test_make_step_reads_a_plain_tensor_attribute_put_in_anothers_place(
    device: str,
) -> None:
    runs = []
    for compiled in (True, False):
        lucid.manual_seed(0)
        model = _Masked().to(device)
        model.mask = model.mask.to(device)
        x, t = _data(device)
        step = make_step(model, F.mse_loss) if compiled else None
        losses = []
        for k in range(3):
            if k == 1:
                model.mask = _values((4,), 0.9, device)
            model.lin.weight.grad = None
            loss = step(x, t) if step is not None else F.mse_loss(model(x), t)
            loss.backward()
            losses.append(float(loss.item()))
        assert model.lin.weight.grad is not None
        runs.append((losses, model.lin.weight.grad.numpy()))
    (losses, grad), (ref_losses, ref_grad) = runs
    np.testing.assert_allclose(losses, ref_losses, rtol=_TOL, atol=_TOL)
    np.testing.assert_allclose(grad, ref_grad, rtol=_TOL, atol=_TOL)


_REBOUND = nn.Identity()
_REBOUND_TENSOR = lucid.zeros(4)


def _reads_rebound(x: lucid.Tensor) -> lucid.Tensor:
    return _REBOUND(x) * _REBOUND_TENSOR


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.parametrize("reach", ["closure", "global"])
@pytest.mark.parametrize("rebound", ["module", "tensor"])
def test_a_function_whose_name_is_bound_to_another_one(
    rebound: str, reach: str
) -> None:
    """A closure cell or a global given another module or tensor moves no tree
    version; what the function reaches is compared on every call instead.
    The two modules have the same structure and so the same key."""
    global _REBOUND, _REBOUND_TENSOR
    first, second = _linear(1, "metal"), _linear(2, "metal")
    scale, other = _values((4,), 0.3, "metal"), _values((4,), 0.8, "metal")
    x, _ = _data("metal")
    if reach == "global":
        _REBOUND, _REBOUND_TENSOR = first, scale
        compiled = lucid.compile(_reads_rebound)
        compiled(x)
        if rebound == "module":
            _REBOUND = second
        else:
            _REBOUND_TENSOR = other
        _close(compiled(x), _reads_rebound(x))
    else:
        module, tensor = first, scale

        def reads(x: lucid.Tensor) -> lucid.Tensor:
            return module(x) * tensor

        compiled = lucid.compile(reads)
        compiled(x)
        if rebound == "module":
            module = second
        else:
            tensor = other
        _close(compiled(x), reads(x))
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def _traces(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every trace a CompiledModule takes from now on."""
    seen: list[object] = []
    real = CompiledModule._compile_for

    def counted(
        self: CompiledModule[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        seen.append(self)
        return real(self, args, kwargs)

    monkeypatch.setattr(CompiledModule, "_compile_for", counted)
    return seen


def _weight_norm(m: nn.Linear) -> nn.Module:
    return nn.utils.weight_norm(m)


def _pruned(m: nn.Linear) -> nn.Module:
    nn.utils.prune.l1_unstructured(m, "weight", amount=0.5)
    return m


def _spectral_norm(m: nn.Linear) -> nn.Module:
    return nn.utils.spectral_norm(m).eval()


_REPARAMETRISED = {
    "weight_norm": _weight_norm,
    "prune": _pruned,
    "spectral_norm-eval": _spectral_norm,
}


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.parametrize("kind", list(_REPARAMETRISED))
def test_a_reparametrised_module_called_eagerly_in_between_is_not_traced_again(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its pre-hook sets ``module.weight`` as a plain attribute on every
    eager forward, but a trace makes that tensor itself and is never fed it:
    the record leaves it out (LCD-606), so an eager call in between retraces
    nothing.  The version is one for every tree, and any setattr moves it —
    here on a module of no concern — so the record is compared each time."""
    lucid.manual_seed(0)
    model = _REPARAMETRISED[kind](nn.Linear(4, 4).to("metal"))
    elsewhere = nn.Identity()
    x, _ = _data("metal")
    traces = _traces(monkeypatch)
    compiled = lucid.compile(model)
    for k in range(3):
        _close(compiled(x), model(x))
        elsewhere.flag = k
    assert len(traces) == 1


class _Stores(nn.Module):
    """Keeps its input as its forward runs, the way a module keeps its last
    activation for a hook or a later read."""

    def __init__(self, store: Callable[[nn.Module, lucid.Tensor], None]) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.store = store

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.store(self, x)
        return self.lin(x)


_LAST: lucid.Tensor | None = None
_STORING_NET: nn.Module = nn.Identity()


def _stores(kind: str) -> Callable[[nn.Module, lucid.Tensor], None]:
    """``forward``'s way of keeping its input: an attribute, a closure variable
    rebound with ``nonlocal``, or a global."""
    last: lucid.Tensor | None = None

    def attribute(m: nn.Module, x: lucid.Tensor) -> None:
        m.last = x

    def closure(m: nn.Module, x: lucid.Tensor) -> None:
        nonlocal last
        last = x

    def module_global(m: nn.Module, x: lucid.Tensor) -> None:
        global _LAST
        _LAST = x

    return {"attribute": attribute, "nonlocal": closure, "global": module_global}[kind]


def _storing_function(
    model: nn.Module, kind: str, reach: str
) -> Callable[[lucid.Tensor], lucid.Tensor]:
    """A function over ``model`` that keeps the input itself — in its own
    closure cell or a global — when ``kind`` says so."""
    global _STORING_NET
    _STORING_NET = model
    last: lucid.Tensor | None = None

    def through_closure(x: lucid.Tensor) -> lucid.Tensor:
        nonlocal last
        global _LAST
        if kind == "nonlocal":
            last = x
        elif kind == "global":
            _LAST = x
        return model(x)

    def through_global(x: lucid.Tensor) -> lucid.Tensor:
        nonlocal last
        global _LAST
        if kind == "nonlocal":
            last = x
        elif kind == "global":
            _LAST = x
        return _STORING_NET(x)

    return through_closure if reach == "closure" else through_global


@_METAL
@pytest.mark.parametrize(
    "entry", ["module", "bound-method", "closure", "global", "make_step"]
)
@pytest.mark.parametrize("kind", ["attribute", "nonlocal", "global"])
def test_a_forward_that_keeps_its_input_is_traced_once(
    kind: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slot is the executable's state only if it held the fed tensor before
    the trace.  One the forward assigned the argument to while traced held
    the argument then, and recorded as state it made every new input a
    replacement: 20 compiles in 20 calls (LCD-606 review)."""
    from lucid.compile._entry import step as step_module

    lucid.manual_seed(0)
    # A compiled module keeps it in its forward; a compiled function over the
    # model in its own cell or global, unless it is an attribute of the model.
    own = entry in ("module", "bound-method", "make_step") or kind == "attribute"
    model = _Stores(_stores(kind) if own else lambda m, x: None).to("metal")
    traces = _traces(monkeypatch)
    steps: list[object] = []
    real_step = step_module._compile_step_entry

    def counted_step(*a: object, **k: object) -> object:
        steps.append(a)
        return real_step(*a, **k)

    monkeypatch.setattr(step_module, "_compile_step_entry", counted_step)
    if entry == "make_step":
        run = make_step(model, F.mse_loss)
        t = _values((8, 4), 0.2, "metal")
        for k in range(4):
            x = _values((8, 4), 0.1 * (k + 1), "metal")
            loss = run(x, t)
            _close(loss, F.mse_loss(model(x), t))
        assert len(steps) == 1
        return
    compiled = (
        lucid.compile(model)
        if entry == "module"
        else lucid.compile(model.forward)
        if entry == "bound-method"
        else lucid.compile(_storing_function(model, kind, entry))
    )
    with lucid.no_grad():
        for k in range(4):
            x = _values((8, 4), 0.1 * (k + 1), "metal")
            _close(compiled(x), model.lin(x))
    assert len(traces) == 1


# ── What is still not seen — each its own card ─────────────────────────────

_HELD_IN_A_DICT: dict[str, nn.Module] = {}
_REACHED_BY_A_HELPER: nn.Module = nn.Identity()


def _through_a_dict(x: lucid.Tensor) -> lucid.Tensor:
    return _HELD_IN_A_DICT["model"](x)


def _helper(x: lucid.Tensor) -> lucid.Tensor:
    return _REACHED_BY_A_HELPER(x)


def _through_a_helper(x: lucid.Tensor) -> lucid.Tensor:
    return _helper(x)


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.parametrize("reach", ["dict-held", "helper-reached"])
@pytest.mark.parametrize("change", ["parameter-setattr", "rebound"])
def test_a_module_reached_less_directly(reach: str, change: str) -> None:
    """A module held in a dict global, or reached through a function the
    compiled one calls, is watched the same way a direct closure cell or
    global is (LCD-642): both a parameter swapped on it and the name itself
    rebound to another module force the next compiled call to match eager."""
    global _REACHED_BY_A_HELPER
    first, second = _linear(1, "metal"), _linear(2, "metal")
    _HELD_IN_A_DICT["model"] = _REACHED_BY_A_HELPER = first
    fn = _through_a_dict if reach == "dict-held" else _through_a_helper
    compiled = lucid.compile(fn)
    x, _ = _data("metal")
    compiled(x)
    if change == "parameter-setattr":
        first.weight = nn.Parameter(_values((4, 4), 0.3, "metal"))
    else:
        _HELD_IN_A_DICT["model"] = _REACHED_BY_A_HELPER = second
    _close(compiled(x), fn(x))


@pytest.mark.usefixtures("inference")
@_METAL
@pytest.mark.xfail(
    strict=True,
    reason="LCD-598: p.data = v gives the parameter new storage without "
    "moving the tree version",
)
def test_a_parameter_given_new_storage_through_data() -> None:
    model = _net("metal")
    x, _ = _data("metal")
    compiled = lucid.compile(model)
    compiled(x)
    model.inp.weight.data = _values((4, 4), 0.3, "metal")
    _close(compiled(x), model(x))
