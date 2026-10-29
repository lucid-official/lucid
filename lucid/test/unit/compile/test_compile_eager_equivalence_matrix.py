"""Every compiled entry point is the eager program, model by model.

Specific compile bugs are pinned in their own files.  This module asks the
generic question instead: take a model a user would actually train — an MLP,
a conv net with batch norm, a transformer encoder layer, a small GPT-2 — and
run it through each compiled entry point

    ``lucid.compile(model)``                      forward
    ``make_step`` / ``compiled_step``             compiled forward + backward, eager optimizer
    ``fused_step``                                forward + backward + optimizer in one executable

next to the eager loop that does the same thing.  What a user can observe must
agree: the output, the loss on each of five steps, the parameters and buffers
afterwards, and the optimizer's ``state_dict`` (SGD with momentum, AdamW).  A
compiled decode step over a ``StaticCache`` is held to the logits of a
full-sequence forward, which needs no cache at all.

Tolerances are loose enough for a different Apple GPU (CI's is not this one):
MPSGraph differentiates and fuses on its own, so a handful of ulp per step is
expected, and the bound grows with depth.  Nothing here accepts a different
*decision* — shapes, dtypes, state keys and step counters must be identical.

Compile is Metal-only: CPU traces fall back to eager, so there is no CPU row.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile as compile_module
from lucid.compile import compiled_step, fused_step, make_step
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

pytestmark = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")

STEPS = 5
_COMPILED_PATHS = ["make_step", "compiled_step", "fused_step"]
_VOCAB = 48


# ── models ───────────────────────────────────────────────────────────────────


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).relu())


class _Tied(nn.Module):
    """Two layers sharing one weight — a tied parameter reached twice."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.b.weight = self.a.weight
        self.head = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.b(self.a(x).tanh()).tanh())


class _ConvBN(nn.Module):
    """Conv + batch norm + pool + linear: parameters *and* running-stat buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(F.max_pool2d(self.bn(self.conv(x)).relu(), 2).flatten(1))


class _Recurrent(nn.Module):
    """An embedding with a padding row, a GRU and a layer norm — the last step's state."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(_VOCAB, 8, padding_idx=0)
        self.gru = nn.GRU(8, 12, batch_first=True)
        self.norm = nn.LayerNorm(12)
        self.fc = nn.Linear(12, 4)

    def forward(self, ids: lucid.Tensor) -> lucid.Tensor:
        seq, _ = self.gru(self.emb(ids))
        return self.fc(self.norm(seq[:, -1]))


class _Decoder(nn.Module):
    """Transposed conv + group norm + upsample + SiLU: an image-to-image net."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Conv2d(2, 4, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(2, 4)
        self.up = nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1)
        self.out = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = F.silu(self.norm(self.down(x)))
        return self.out(F.interpolate(self.up(h), scale_factor=1.0, mode="nearest"))


class _Encoder(nn.Module):
    """One post-norm transformer encoder layer, no dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            16, 2, dim_feedforward=32, dropout=0.0, batch_first=True
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.layer(x)


class _GPT2Logits(nn.Module):
    """A small GPT-2 language model that returns its logits (a plain tensor)."""

    def __init__(self) -> None:
        super().__init__()
        self.lm = _small_gpt2()

    def forward(self, ids: lucid.Tensor) -> lucid.Tensor:
        return self.lm(ids).logits


def _small_gpt2() -> nn.Module:
    return M.create_model(
        "gpt2_small_lm",
        vocab_size=_VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _mse(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    # Module-level: ``fused_step`` keys its cache on the loss function's identity.
    return F.mse_loss(out, target)


def _next_token(logits: lucid.Tensor, ids: lucid.Tensor) -> lucid.Tensor:
    return F.cross_entropy(logits.reshape(-1, _VOCAB), ids.reshape(-1))


Batch = Callable[[int], tuple[lucid.Tensor, lucid.Tensor]]


@dataclass(frozen=True)
class _Spec:
    """A model under test: how to build it, feed it, score it, and how far a run may drift."""

    model: Callable[[], nn.Module]
    data: Batch
    loss: Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]
    tol: float  # bounds loss / parameter / state drift after ``STEPS`` steps


def _regression(shape: tuple[int, ...], out: tuple[int, ...]) -> Batch:
    def make(seed: int) -> tuple[lucid.Tensor, lucid.Tensor]:
        lucid.manual_seed(seed)
        return (
            lucid.randn(*shape, device=COMPILE_DEVICE),
            lucid.randn(*out, device=COMPILE_DEVICE),
        )

    return make


def _tokens(seed: int) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(seed)
    ids = lucid.randint(0, _VOCAB, (4, 12), dtype=lucid.int64, device=COMPILE_DEVICE)
    return ids, ids


def _class_tokens(seed: int) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(seed)
    ids = lucid.randint(0, _VOCAB, (6, 7), dtype=lucid.int64, device=COMPILE_DEVICE)
    ids[:, 0] = 0  # padding positions
    return ids, lucid.randint(0, 4, (6,), dtype=lucid.int64, device=COMPILE_DEVICE)


def _cross_entropy_labels(logits: lucid.Tensor, labels: lucid.Tensor) -> lucid.Tensor:
    return F.cross_entropy(logits, labels)


_MODELS: dict[str, _Spec] = {
    "mlp": _Spec(_MLP, _regression((16, 8), (16, 4)), _mse, 1e-5),
    "tied": _Spec(_Tied, _regression((16, 8), (16, 4)), _mse, 1e-5),
    "conv_bn": _Spec(_ConvBN, _regression((8, 2, 8, 8), (8, 4)), _mse, 1e-4),
    "encoder": _Spec(_Encoder, _regression((4, 6, 16), (4, 6, 16)), _mse, 1e-4),
    "recurrent": _Spec(_Recurrent, _class_tokens, _cross_entropy_labels, 1e-4),
    "decoder": _Spec(_Decoder, _regression((4, 2, 8, 8), (4, 2, 8, 8)), _mse, 1e-4),
    "gpt2": _Spec(_GPT2Logits, _tokens, _next_token, 1e-4),
}

_OPTIMIZERS: dict[str, Callable[[nn.Module], optim.Optimizer]] = {
    "sgd_momentum": lambda m: optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
    # A large eps keeps Adam from turning a gradient that is round-off in one
    # path and exactly zero in the other (the bias in front of a batch norm)
    # into a full-size step: ``m / (sqrt(v) + eps)`` is otherwise ill-conditioned there.
    "sgd_nesterov_decay": lambda m: optim.SGD(
        m.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-2
    ),
    "adamw": lambda m: optim.AdamW(m.parameters(), lr=1e-2, eps=1e-3),
}


def _build(name: str) -> nn.Module:
    lucid.manual_seed(7)
    return _MODELS[name].model().to(COMPILE_DEVICE)


# ── observing a run ──────────────────────────────────────────────────────────


def _flat(obj: object) -> list[np.ndarray]:
    """Every tensor / number of a nested state, as float64 arrays, in order."""
    if isinstance(obj, lucid.Tensor):
        return [obj.detach().to(lucid.float32).numpy().astype(np.float64)]
    if isinstance(obj, dict):
        return [a for v in obj.values() for a in _flat(v)]
    if isinstance(obj, (list, tuple)):
        return [a for v in obj for a in _flat(v)]
    if isinstance(obj, (int, float)):
        return [np.asarray(obj, dtype=np.float64)]
    return []


def _keys(obj: object) -> object:
    """The shape of a nested state with every leaf replaced by its dtype / type."""
    if isinstance(obj, lucid.Tensor):
        return (str(obj.dtype), tuple(obj.shape))
    if isinstance(obj, dict):
        return {k: _keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_keys(v) for v in obj]
    return type(obj).__name__


def _assert_close(got: list[np.ndarray], want: list[np.ndarray], tol: float, what: str) -> None:
    assert len(got) == len(want), f"{what}: {len(got)} leaves vs {len(want)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g.shape == w.shape, f"{what}[{i}]: shape {g.shape} vs {w.shape}"
        scale = max(1.0, float(np.abs(w).max(initial=0.0)))
        drift = float(np.abs(g - w).max(initial=0.0))
        assert drift <= tol * scale, f"{what}[{i}]: drift {drift:.3e} > {tol * scale:.3e}"


class _Run:
    """One training run of ``path`` over a fresh seeded model; records the losses."""

    def __init__(self, name: str, opt: str, path: str, *, frozen: tuple[str, ...] = ()) -> None:
        self.model = _build(name)
        for pname, p in self.model.named_parameters():
            if pname.startswith(frozen):
                p.requires_grad_(False)
        self.opt = _OPTIMIZERS[opt](self.model)
        self.loss_fn = _MODELS[name].loss
        self.path = path
        self.losses: list[float] = []
        if path == "make_step":
            self._step = make_step(self.model, self.loss_fn)
        elif path == "fused_step":
            self._step = fused_step(self.model, self.loss_fn, self.opt)

    def step(self, x: lucid.Tensor, t: lucid.Tensor) -> None:
        if self.path == "fused_step":
            self.losses.append(float(self._step(x, t).item()))
            return
        self.opt.zero_grad()
        if self.path == "eager":
            loss = self.loss_fn(self.model(x), t)
            loss.backward()
        elif self.path == "make_step":
            loss = self._step(x, t)
            loss.backward()
        else:
            loss = compiled_step(self.model, x, lambda y: self.loss_fn(y, t))
        self.losses.append(float(loss.item()))
        self.opt.step()

    def state(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            _flat(dict(self.model.named_parameters())),
            _flat(dict(self.model.named_buffers())),
            _flat(self.opt.state_dict()),
        )


def _train(
    name: str, opt: str, path: str, *, frozen: tuple[str, ...] = (), seeds: tuple[int, ...] = ()
) -> _Run:
    run = _Run(name, opt, path, frozen=frozen)
    for seed in seeds or tuple(100 + k for k in range(STEPS)):
        run.step(*_MODELS[name].data(seed))
    return run


# ── training steps ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("opt", list(_OPTIMIZERS))
@pytest.mark.parametrize("path", _COMPILED_PATHS)
@pytest.mark.parametrize("name", list(_MODELS))
def test_a_compiled_run_is_the_eager_run(name: str, path: str, opt: str) -> None:
    _assert_runs_agree(name, opt, path)


def _assert_runs_agree(
    name: str, opt: str, path: str, *, frozen: tuple[str, ...] = (), seeds: tuple[int, ...] = ()
) -> tuple[_Run, _Run]:
    tol = _MODELS[name].tol
    eager = _train(name, opt, "eager", frozen=frozen, seeds=seeds)
    comp = _train(name, opt, path, frozen=frozen, seeds=seeds)

    _assert_close(
        [np.asarray(comp.losses)], [np.asarray(eager.losses)], tol, f"{name}/{path} losses"
    )
    p_c, b_c, s_c = comp.state()
    p_e, b_e, s_e = eager.state()
    start = _flat(dict(_build(name).named_parameters()))
    assert any(np.abs(a - b).max() > 1e-3 for a, b in zip(p_e, start, strict=True)), "no training"
    _assert_close(p_c, p_e, tol, "parameters")
    _assert_close(b_c, b_e, tol, "buffers")
    # The optimizer state_dict has the same keys, dtypes, shapes and step counts.
    assert _keys(comp.opt.state_dict()) == _keys(eager.opt.state_dict())
    _assert_close(s_c, s_e, tol, "optimizer state_dict")
    return eager, comp


@pytest.mark.parametrize("path", _COMPILED_PATHS)
def test_a_frozen_layer_stays_put(path: str) -> None:
    # The frozen layer is still in the optimizer; it has no gradient, so the
    # step must leave it — values and optimizer state — as eager does.
    eager, comp = _assert_runs_agree("mlp", "adamw", path, frozen=("fc1",))
    start = _flat(dict(_build("mlp").named_parameters()))[:2]
    _assert_close(_flat(dict(comp.model.named_parameters()))[:2], start, 0.0, "frozen fc1")


@pytest.mark.parametrize("path", _COMPILED_PATHS)
def test_a_changing_batch_size_is_the_eager_run(path: str) -> None:
    # Distinct shapes are distinct executables; returning to one must reuse it
    # without carrying anything over from the other.
    _assert_runs_agree("mlp", "sgd_momentum", path, seeds=(100, 101, 102, 101, 100))


@pytest.mark.parametrize("path", _COMPILED_PATHS)
def test_a_trained_model_evaluates_like_the_eager_one(path: str) -> None:
    # What training leaves in the model is what every later reader sees: the
    # eager call, a freshly compiled forward and a state_dict round trip.
    eager, comp = _assert_runs_agree("encoder", "adamw", path)
    x, _ = _MODELS["encoder"].data(900)
    with lucid.no_grad():
        want = eager.model.eval()(x)
        _assert_close(_flat(comp.model.eval()(x)), _flat(want), 1e-4, "eager call after training")
        _assert_close(
            _flat(compile_module(comp.model)(x)), _flat(want), 1e-4, "compiled call after training"
        )
        twin = _build("encoder").eval()
        twin.load_state_dict(comp.model.state_dict())
        _assert_close(_flat(twin(x)), _flat(want), 1e-4, "state_dict round trip")


# ── accumulation, clipping and half precision ────────────────────────────────


def _mlp_loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(out, target)


def _loop(
    path: str, dtype: lucid.dtype, *, accum: int, clip: float | None
) -> tuple[list[float], list[np.ndarray]]:
    """Five optimizer steps of a small MLP; each is ``accum`` micro-batches, then a clip."""
    model = _build("mlp").to(dtype)
    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    fused = fused_step(model, _mlp_loss, opt) if path == "fused_step" else None
    step = make_step(model, _mlp_loss) if path == "make_step" else None
    losses = []
    for k in range(STEPS):
        opt.zero_grad()
        for a in range(accum):
            x, t = (v.to(dtype) for v in _MODELS["mlp"].data(100 + k * 10 + a))
            if fused is not None:
                loss = fused(x, t)
            elif step is not None:
                loss = step(x, t)
                loss.backward()
            elif path == "compiled_step":
                loss = compiled_step(model, x, lambda y, t=t: _mlp_loss(y, t))
            else:
                loss = _mlp_loss(model(x), t)
                loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(float(loss.item()))
    return losses, _flat(dict(model.named_parameters()))


@pytest.mark.parametrize("path", ["make_step", "compiled_step"])
@pytest.mark.parametrize(
    "dtype, tol",
    [(lucid.float32, 1e-5), (lucid.float16, 5e-3), (lucid.bfloat16, 3e-2)],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize("accum, clip", [(1, None), (3, None), (2, 0.1)])
def test_accumulation_clipping_and_half_precision_follow_eager(
    path: str, dtype: lucid.dtype, tol: float, accum: int, clip: float | None
) -> None:
    want_losses, want = _loop("eager", dtype, accum=accum, clip=clip)
    got_losses, got = _loop(path, dtype, accum=accum, clip=clip)
    _assert_close([np.asarray(got_losses)], [np.asarray(want_losses)], tol, "losses")
    _assert_close(got, want, tol, "parameters")


@pytest.mark.parametrize(
    "dtype, tol", [(lucid.float16, 5e-3), (lucid.bfloat16, 3e-2)], ids=["fp16", "bf16"]
)
def test_a_half_precision_fused_step_follows_eager(dtype: lucid.dtype, tol: float) -> None:
    want_losses, want = _loop("eager", dtype, accum=1, clip=None)
    got_losses, got = _loop("fused_step", dtype, accum=1, clip=None)
    _assert_close([np.asarray(got_losses)], [np.asarray(want_losses)], tol, "losses")
    _assert_close(got, want, tol, "parameters")


# ── inputs that are views ────────────────────────────────────────────────────


def _input_layouts() -> dict[str, lucid.Tensor]:
    lucid.manual_seed(1)
    base = lucid.randn(8, 16, device=COMPILE_DEVICE)
    wide = lucid.randn(32, 16, device=COMPILE_DEVICE)
    row = lucid.randn(1, 8, device=COMPILE_DEVICE)
    cube = lucid.randn(8, 2, 8, device=COMPILE_DEVICE)
    return {
        "transposed": base.t(),
        "row_step": wide[::2, :8],
        "column_slice": wide[:16, 4:12],
        "expanded": row.expand(16, 8),
        "permuted": cube.permute(1, 0, 2).reshape(16, 8),
    }


@pytest.mark.parametrize("path", ["forward", "make_step", "fused_step"])
@pytest.mark.parametrize("layout", list(_input_layouts()))
def test_an_input_that_is_a_view_is_read_through_its_strides(layout: str, path: str) -> None:
    # A compiled graph is fed the tensor's elements, not its storage: a
    # transposed, strided or broadcast input must mean what it means to eager.
    x = _input_layouts()[layout]
    lucid.manual_seed(2)
    target = lucid.randn(16, 4, device=COMPILE_DEVICE)

    eager, comp = _build("mlp"), _build("mlp")
    if path == "forward":
        with lucid.no_grad():
            _assert_close(_flat(compile_module(comp)(x)), _flat(eager(x)), 1e-5, "output")
        return

    eager_opt = optim.SGD(eager.parameters(), lr=0.1)
    comp_opt = optim.SGD(comp.parameters(), lr=0.1)
    eager_opt.zero_grad()
    want = _mse(eager(x), target)
    want.backward()
    eager_opt.step()
    if path == "fused_step":
        got = fused_step(comp, _mse, comp_opt)(x, target)
    else:
        comp_opt.zero_grad()
        got = make_step(comp, _mse)(x, target)
        got.backward()
        comp_opt.step()
    _assert_close(_flat(got), _flat(want), 1e-5, "loss")
    _assert_close(
        _flat(dict(comp.named_parameters())), _flat(dict(eager.named_parameters())), 1e-5, "params"
    )


# ── forward ──────────────────────────────────────────────────────────────────


@pytest.mark.filterwarnings(
    "ignore:lucid.compile:lucid.compile._core.fallback.CompileFallbackWarning"
)
@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("name", list(_MODELS))
def test_a_compiled_forward_is_the_eager_forward(name: str, mode: str) -> None:
    tol = _MODELS[name].tol
    eager_model, comp_model = _build(name), _build(name)
    getattr(eager_model, mode)()
    getattr(comp_model, mode)()
    compiled = compile_module(comp_model)
    with lucid.no_grad():
        for k in range(3):
            x, _ = _MODELS[name].data(200 + k)
            want, got = eager_model(x), compiled(x)
            assert tuple(got.shape) == tuple(want.shape) and got.dtype == want.dtype
            _assert_close(_flat(got), _flat(want), tol, f"{name}/{mode} output")
    # A train-mode forward updates running statistics exactly as eager does.
    _assert_close(
        _flat(dict(comp_model.named_buffers())),
        _flat(dict(eager_model.named_buffers())),
        tol,
        "buffers after forward",
    )


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("name", ["mlp", "encoder"])
def test_a_half_precision_compiled_forward_is_the_eager_forward(
    name: str, dtype: lucid.dtype
) -> None:
    # Half precision rounds at every op; a compiled graph may fuse where eager
    # rounds, so the bound is a few half-precision ulp, not float32 round-off.
    eager_model, comp_model = _build(name).to(dtype).eval(), _build(name).to(dtype).eval()
    compiled = compile_module(comp_model)
    x, _ = _MODELS[name].data(300)
    x = x.to(dtype)
    with lucid.no_grad():
        want, got = eager_model(x), compiled(x)
    assert got.dtype == dtype == want.dtype
    _assert_close(_flat(got), _flat(want), 5e-2 if dtype == lucid.bfloat16 else 1e-2, "half")


# ── decode ───────────────────────────────────────────────────────────────────


@lucid.no_grad()
def test_compiled_decode_matches_a_full_sequence_forward() -> None:
    # The oracle needs no cache: the logits for position t of the whole prefix.
    lucid.manual_seed(5)
    gpt = _small_gpt2().to(COMPILE_DEVICE).eval()
    prompt = [5, 9, 2, 7]
    feed = [11, 3, 40, 22, 8]
    cache = StaticCache(32)
    dev = COMPILE_DEVICE

    gpt(
        lucid.tensor([prompt], device=dev).long(),
        past_key_values=cache,
        use_cache=True,
        cache_position=lucid.arange(0, len(prompt), device=dev).long(),
    )

    def fwd(tok: lucid.Tensor, pos: lucid.Tensor) -> lucid.Tensor:
        return gpt(tok, past_key_values=cache, use_cache=True, cache_position=pos).logits

    step = compiled_decode_step(fwd, cache)
    seq = list(prompt)
    for i, t in enumerate(feed):
        got = step(
            lucid.tensor([[t]], device=dev).long(),
            lucid.tensor([len(prompt) + i], device=dev).long(),
        )[:, -1]
        seq.append(t)
        want = gpt(lucid.tensor([seq], device=dev).long()).logits[:, -1]
        assert tuple(got.shape) == tuple(want.shape)
        _assert_close(_flat(got), _flat(want), 1e-4, f"decode logits at token {i}")
        assert int(got.argmax(-1).item()) == int(want.argmax(-1).item())
    assert len(step.cache) == 1, "one decode executable serves every token"
