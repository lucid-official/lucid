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
from lucid.compile import compile_optimizer, fused_step, make_step
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


# When the replacement comes: the step to make it before.  Before the first
# call matters too — ``make_step`` takes the parameters, and the segmented step
# its children, when it is built.
_WHEN = {"before-the-first-call": 0, "after-the-first-call": 2}


def _compiled(model: _Net, entry: str) -> Callable[[lucid.Tensor], lucid.Tensor]:
    """``lucid.compile`` of the model, of its bound forward, or of a closure."""
    if entry == "module":
        return lucid.compile(model)
    if entry == "bound-method":
        return lucid.compile(model.forward)

    def call(x: lucid.Tensor) -> lucid.Tensor:
        return model(x)

    return lucid.compile(call)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("when", list(_WHEN))
@pytest.mark.parametrize("entry", ["module", "bound-method", "closure"])
@pytest.mark.parametrize("change", list(_CHANGES))
def test_a_compiled_module_answers_with_what_the_model_holds_now(
    change: str, entry: str, when: str, device: str
) -> None:
    model = _net(device)
    x, _ = _data(device)
    compiled = _compiled(model, entry)
    for _ in range(_WHEN[when]):
        _close(compiled(x), model(x))
    _CHANGES[change](model, device)
    _close(compiled(x), model(x))
    if device == "metal":
        # It answered compiled, traced again — not eagerly.
        info = compiled.cache_info()  # type: ignore[attr-defined]
        assert info["entries"] == 1 and not info["eager_only"], info


def _named(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in model.named_parameters()}


def _assert_compiled(cache: object, eager_only: object) -> None:
    """A step that answered eagerly would match eager without being fixed."""
    assert cache and not eager_only, (cache, eager_only)


def _make_step_run(
    model: _Net, change: str, at: int, device: str, compiled: bool
) -> list[float]:
    x, t = _data(device)
    step = make_step(model, F.mse_loss) if compiled else None
    opt = optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for k in range(4):
        if k == at:
            _CHANGES[change](model, device)
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
    losses = _make_step_run(model, change, _WHEN[when], device, compiled=True)
    ref_losses = _make_step_run(ref, change, _WHEN[when], device, compiled=False)
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
        if k == _WHEN[when]:
            for m, o in ((model, opt), (ref, ref_opt)):
                _CHANGES[change](m, device)
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
        if k == _WHEN[when]:
            _SEQ_CHANGES[change](model, device)
            _SEQ_CHANGES[change](ref, device)
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


# ── What is still not seen — each its own card ─────────────────────────────


class _Masked(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        # A plain attribute, not a buffer: it lives in ``__dict__``.
        self.mask = lucid.ones(4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.lin(x) * self.mask


@_METAL
@pytest.mark.xfail(
    strict=True,
    reason="LCD-560 follow-up: a plain tensor attribute the forward reads is "
    "not in the tree record (only the ones the trace fed may go in: the "
    "reparametrisations rewrite theirs on every eager forward)",
)
def test_a_plain_tensor_attribute_put_in_anothers_place() -> None:
    lucid.manual_seed(0)
    model = _Masked().to("metal")
    model.mask = model.mask.to("metal")
    x, _ = _data("metal")
    compiled = lucid.compile(model)
    compiled(x)
    model.mask = _values((4,), 0.9, "metal")
    _close(compiled(x), model(x))


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


# A module the compiled function reaches through a module-level name, not a
# closure cell or a bound ``self``: nothing names it to the compiled wrapper.
_GLOBALS: dict[str, nn.Sequential] = {}


def _global_forward(x: lucid.Tensor) -> lucid.Tensor:
    return _GLOBALS["model"](x)


_GLOBAL_CHANGES: dict[str, Callable[[nn.Sequential], object]] = {
    "parameter-setattr": lambda m: setattr(
        m[0], "weight", nn.Parameter(_values((4, 4), 0.3, "metal"))
    ),
    "submodule": lambda m: m.__setitem__(1, nn.Sigmoid()),
    "eval": lambda m: m.eval(),
}


@_METAL
@pytest.mark.xfail(
    strict=True,
    reason="LCD-599: a module reached through a module-level global is not "
    "watched — not by the key, not by the tree record",
)
@pytest.mark.parametrize("change", list(_GLOBAL_CHANGES))
def test_a_module_reached_through_a_global(change: str) -> None:
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(0.5)).to("metal")
    # Dropout is on only for the change that turns it off.
    model.train(change == "eval")
    _GLOBALS["model"] = model
    x, _ = _data("metal")
    compiled = lucid.compile(_global_forward)
    compiled(x)
    _GLOBAL_CHANGES[change](model)
    _close(compiled(x), model(x))
