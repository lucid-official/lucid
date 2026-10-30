"""Submodule training/scalar-attribute cache keys and module hooks under compile.

Covers GitHub #251 (LCD-178):

1. Submodule ``.training`` flags and public Python scalar attributes across
   ``model.modules()`` are part of :class:`CacheKey`, so flipping
   ``m.drop.eval()`` or mutating ``m.scale = 2.0`` invalidates the cached
   executable in both :class:`CompiledModule` and :func:`make_step`.
2. User module hooks (``_forward_pre_hooks``, ``_forward_hooks``,
   ``_backward_pre_hooks``, ``_backward_hooks``, and global module hooks)
   registered on the root module or any submodule execute on every call with
   live tensors (whether registered before or after compilation), and removing
   the hook handle resumes compiled execution without poisoning ``eager_only``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core.signature import model_has_active_hooks, signature_of
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


class _DropScaleNet(nn.Module):
    def __init__(self, p: float = 0.5, scale: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(p=p)
        self.fc2 = nn.Linear(16, 4)
        self.scale = scale

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.drop(self.fc1(x).relu())
        return self.fc2(h) * self.scale


class _NestedBNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
        self.head = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.block(x))


# ── 1. Submodule .train() / .eval() invalidates cache ───────────────────────


@pytest.mark.usefixtures("inference")
def test_submodule_dropout_eval_flip_retraces_compiled_module() -> None:
    """Flipping ``m.drop.eval()`` without touching ``m.training`` must re-trace."""
    m = _DropScaleNet(p=0.5).to(COMPILE_DEVICE)
    m.train()
    cm = lucid.compile(m)
    x = metal_tensor(8, 8)

    # First call in full train mode compiles train graph (with RNG dropout feed).
    _ = cm(x)
    assert cm.cache_info()["entries"] == 1

    # Flip only the Dropout submodule to eval; root ``m.training`` stays True.
    m.drop.eval()
    assert m.training is True
    assert m.drop.training is False

    y1 = cm(x).numpy()
    y2 = cm(x).numpy()
    want = m(x).numpy()
    assert cm.cache_info()["entries"] == 2
    assert np.allclose(y1, want, rtol=1e-5, atol=1e-5)
    assert np.array_equal(y1, y2), "eval-mode dropout submodule must be deterministic"

    # Flipping back to train reuses the first compiled entry.
    m.drop.train()
    lucid.manual_seed(42)
    got_train = cm(x).numpy()
    lucid.manual_seed(42)
    want_train = m(x).numpy()
    assert cm.cache_info()["entries"] == 2
    assert np.allclose(got_train, want_train, rtol=1e-5, atol=1e-5)


def test_submodule_dropout_eval_flip_retraces_make_step() -> None:
    """Flipping ``m.drop.eval()`` under ``make_step`` re-traces fwd+bwd."""
    m = _DropScaleNet(p=0.5).to(COMPILE_DEVICE)
    m.train()
    x = metal_tensor(8, 8)

    def loss_fn(y: lucid.Tensor) -> lucid.Tensor:
        return (y * y).sum()

    step = lucid.compile.make_step(m, loss_fn)
    step(x).backward()
    assert len(step.cache) == 1

    # Flip only the Dropout child to eval mode.
    m.drop.eval()
    for p in m.parameters():
        p.grad = None
    loss_compiled = step(x)
    loss_compiled.backward()
    assert len(step.cache) == 2
    grads_compiled = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

    for p in m.parameters():
        p.grad = None
    loss_eager = loss_fn(m(x))
    loss_eager.backward()
    grads_eager = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

    assert np.isclose(float(loss_compiled.item()), float(loss_eager.item()), rtol=1e-5)
    for gc, ge in zip(grads_compiled, grads_eager):
        assert np.allclose(gc, ge, rtol=1e-4, atol=1e-5)


def test_nested_batchnorm_submodule_mode_flip_compiled_module_and_step() -> None:
    """Nested BatchNorm mode flips are respected even when root ``m.training`` is unchanged."""
    m = _NestedBNNet().to(COMPILE_DEVICE)
    bn = m.block[1]
    assert isinstance(bn, nn.BatchNorm1d)
    x = metal_tensor(8, 8)

    # Start in full eval mode: CompiledModule compiles batch_norm_eval.
    m.eval()
    cm = lucid.compile(m)
    with lucid.no_grad():
        _ = cm(x)
    assert cm.cache_info()["entries"] == 1
    rm_before = bn.running_mean.numpy().copy()  # type: ignore[union-attr]

    # Put ONLY the nested BN into training mode while root m.training is False.
    # Forward-only CompiledModule must fall back to eager so running_mean updates.
    bn.train()
    assert m.training is False
    with lucid.no_grad():
        _ = cm(x)
    rm_after = bn.running_mean.numpy().copy()  # type: ignore[union-attr]
    assert not np.allclose(rm_before, rm_after), "training BN submodule must update running_mean"

    # Under make_step, flipping bn.eval() <-> bn.train() switches between
    # compiled eval BN (no running stat write-back) and compiled train BN.
    bn.eval()
    step = lucid.compile.make_step(m, lambda y: (y * y).sum())
    rm_frozen = bn.running_mean.numpy().copy()  # type: ignore[union-attr]
    step(x).backward()
    assert np.allclose(bn.running_mean.numpy(), rm_frozen)  # type: ignore[union-attr]

    bn.train()
    step(x).backward()
    assert len(step.cache) == 2
    assert not np.allclose(bn.running_mean.numpy(), rm_frozen)  # type: ignore[union-attr]


@pytest.mark.usefixtures("inference")
def test_callable_closing_over_module_tracks_submodule_mode() -> None:
    """Plain function compiled via ``lucid.compile`` tracks closed-over module state."""
    m = _DropScaleNet(p=0.5).to(COMPILE_DEVICE)
    m.train()

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return m(x)

    compiled_fn = lucid.compile(fn)
    x = metal_tensor(8, 8)
    _ = compiled_fn(x)
    assert compiled_fn.cache_info()["entries"] == 1

    m.drop.eval()
    y1 = compiled_fn(x).numpy()
    y2 = compiled_fn(x).numpy()
    assert compiled_fn.cache_info()["entries"] == 2
    assert np.array_equal(y1, y2)


# ── 2. Dynamic Python scalar attribute mutations ────────────────────────────


def test_scalar_attribute_mutation_retraces_compiled_module_and_step() -> None:
    """Mutating a scalar attribute on ``model`` or a submodule invalidates the cache."""
    m = _DropScaleNet(p=0.0, scale=1.0).to(COMPILE_DEVICE)
    m.eval()
    x = metal_tensor(4, 8)

    cm = lucid.compile(m)
    with lucid.no_grad():
        y_one = cm(x).numpy()
        assert cm.cache_info()["entries"] == 1

        # Mutate root scalar attribute.
        m.scale = 2.5
        y_scaled = cm(x).numpy()
        assert cm.cache_info()["entries"] == 2
        assert np.allclose(y_scaled, y_one * 2.5, rtol=1e-5, atol=1e-5)

        # Mutate submodule scalar attribute (Dropout.p).
        m.train()
        m.drop.p = 0.0
        _ = cm(x)
        entries_before = cm.cache_info()["entries"]
        m.drop.p = 0.5
        _ = cm(x)  # first call with p=0.5 traces + compiles
        assert cm.cache_info()["entries"] == entries_before + 1
        lucid.manual_seed(7)
        y_drop = cm(x).numpy()
        lucid.manual_seed(7)
        want_drop = m(x).numpy()
        assert np.allclose(y_drop, want_drop, rtol=1e-5, atol=1e-5)

    # Also verify make_step re-traces on scalar attribute change.
    m.eval()
    m.scale = 1.0
    step = lucid.compile.make_step(m, lambda y: (y * y).sum())
    l1 = float(step(x).item())
    m.scale = 3.0
    l2 = float(step(x).item())
    assert len(step.cache) == 2
    assert np.isclose(l2, l1 * 9.0, rtol=1e-4)


def test_signature_scalar_attr_edge_cases() -> None:
    """Attribute fingerprinting distinguishes types/NaN/-0.0, follows short
    containers changed in place (LCD-704) and ignores what is not state."""
    m = nn.Linear(4, 4).to(COMPILE_DEVICE)
    x = metal_tensor(2, 4)

    m.flag = 1  # type: ignore[attr-defined]
    k_int = signature_of(m, (x,), {}, dynamic=False)
    m.flag = 1.0  # type: ignore[attr-defined]
    k_float = signature_of(m, (x,), {}, dynamic=False)
    m.flag = True  # type: ignore[attr-defined]
    k_bool = signature_of(m, (x,), {}, dynamic=False)
    assert len({k_int, k_float, k_bool}) == 3

    m.flag = 0.0  # type: ignore[attr-defined]
    k_pos_zero = signature_of(m, (x,), {}, dynamic=False)
    m.flag = -0.0  # type: ignore[attr-defined]
    k_neg_zero = signature_of(m, (x,), {}, dynamic=False)
    assert k_pos_zero != k_neg_zero

    m.flag = float("nan")  # type: ignore[attr-defined]
    assert signature_of(m, (x,), {}, dynamic=False) == signature_of(m, (x,), {}, dynamic=False)

    # A list or dict of scalars is state a forward can branch on, changed in
    # place or not; one holding a tensor is data the trace is fed, not state.
    m.history = [1, 2, 3]  # type: ignore[attr-defined]
    m.meta = {"a": 1}  # type: ignore[attr-defined]
    held = [x]
    setattr(m, "held", held)
    k_with_containers = signature_of(m, (x,), {}, dynamic=False)
    m.history.append(4)  # type: ignore[attr-defined]
    k_appended = signature_of(m, (x,), {}, dynamic=False)
    assert k_appended != k_with_containers
    held.append(metal_tensor(2, 4))
    assert signature_of(m, (x,), {}, dynamic=False) == k_appended


# ── 3. Module hooks under CompiledModule and make_step ──────────────────────


@pytest.mark.usefixtures("inference")
def test_forward_hook_on_submodule_before_and_after_compile() -> None:
    """Submodule forward hooks run once per call with live tensors before & after compile."""
    m = _DropScaleNet(p=0.0).to(COMPILE_DEVICE)
    m.eval()
    x = metal_tensor(4, 8)

    recorded: list[np.ndarray] = []

    def _record_hook(mod: nn.Module, inp: tuple[lucid.Tensor, ...], out: lucid.Tensor) -> None:
        recorded.append(out.numpy().copy())

    # 1. Hook registered BEFORE compile: must fire exactly once per call.
    h1 = m.fc1.register_forward_hook(_record_hook)
    cm = lucid.compile(m)
    out1 = cm(x)
    assert len(recorded) == 1
    want_out1 = m(x).numpy()
    assert len(recorded) == 2  # second increment was from eager m(x) above
    assert np.allclose(recorded[0], recorded[1], rtol=1e-5, atol=1e-5)
    assert np.allclose(out1.numpy(), want_out1, rtol=1e-5, atol=1e-5)

    # Removing the hook allows compilation to proceed and cache an executable!
    h1.remove()
    assert not model_has_active_hooks(m)
    _ = cm(x)
    _ = cm(x)
    assert len(recorded) == 2  # hook no longer called
    assert cm.cache_info()["entries"] == 1
    assert not cm.cache_info()["eager_only"]
    assert cm.timing()[0]["n_hits"] >= 2

    # 2. Hook registered AFTER compile (when key is already in cm._cache):
    # must still execute on every call, and removing it resumes cache hits.
    h2 = m.fc1.register_forward_hook(_record_hook)
    x2 = metal_tensor(4, 8)
    _ = cm(x2)
    assert len(recorded) == 3
    h2.remove()
    assert np.allclose(recorded[-1], m.fc1(x2).numpy(), rtol=1e-5, atol=1e-5)

    hits_before = cm.timing()[0]["n_hits"]
    _ = cm(x2)
    assert cm.timing()[0]["n_hits"] == hits_before + 1
    assert len(recorded) == 3


def test_forward_pre_hook_and_output_transform_hook() -> None:
    """Forward pre-hooks and output-modifying forward hooks match eager in compile & step."""
    m = _DropScaleNet(p=0.0).to(COMPILE_DEVICE)
    m.eval()
    x = metal_tensor(4, 8)

    cm = lucid.compile(m)
    step = lucid.compile.make_step(m, lambda y: (y * y).sum())

    # Warm up compiled caches before attaching hooks.
    with lucid.no_grad():
        _ = cm(x)
    step(x).backward()
    assert cm.cache_info()["entries"] == 1
    assert len(step.cache) == 1

    # Pre-hook scales input to fc1 by 2.0; post-hook adds 1.0 to fc2 output.
    h_pre = m.fc1.register_forward_pre_hook(lambda mod, args: (args[0] * 2.0,))
    h_post = m.fc2.register_forward_hook(lambda mod, args, out: out + 1.0)

    try:
        with lucid.no_grad():
            got_cm = cm(x).numpy()
        want_cm = m(x).numpy()
        assert np.allclose(got_cm, want_cm, rtol=1e-5, atol=1e-5)

        for p in m.parameters():
            p.grad = None
        loss_step = step(x)
        loss_step.backward()
        grads_step = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

        for p in m.parameters():
            p.grad = None
        loss_eager = (m(x) ** 2).sum()
        loss_eager.backward()
        grads_eager = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

        assert np.isclose(float(loss_step.item()), float(loss_eager.item()), rtol=1e-5)
        for gs, ge in zip(grads_step, grads_eager):
            assert np.allclose(gs, ge, rtol=1e-4, atol=1e-5)
    finally:
        h_pre.remove()
        h_post.remove()


def test_backward_hooks_on_submodule_under_make_step() -> None:
    """Backward pre-hooks and full backward hooks on a submodule run and transform grads."""
    m = _DropScaleNet(p=0.0).to(COMPILE_DEVICE)
    m.eval()
    x = metal_tensor(4, 8)

    def loss_fn(y: lucid.Tensor) -> lucid.Tensor:
        return (y * y).sum()

    step = lucid.compile.make_step(m, loss_fn)
    # Populate compiled step cache first.
    step(x).backward()
    assert len(step.cache) == 1

    bwd_pre_seen: list[int] = []
    bwd_seen: list[int] = []

    def _bwd_pre(
        mod: nn.Module, grad_output: tuple[lucid.Tensor | None, ...]
    ) -> tuple[lucid.Tensor | None, ...]:
        bwd_pre_seen.append(len(grad_output))
        return tuple(g * 3.0 if g is not None else None for g in grad_output)

    def _bwd_full(
        mod: nn.Module,
        grad_input: tuple[lucid.Tensor | None, ...],
        grad_output: tuple[lucid.Tensor | None, ...],
    ) -> tuple[lucid.Tensor | None, ...]:
        bwd_seen.append(len(grad_input))
        return tuple(g * 2.0 if g is not None else None for g in grad_input)

    h_bpre = m.fc2.register_full_backward_pre_hook(_bwd_pre)
    h_bfull = m.fc2.register_full_backward_hook(_bwd_full)

    try:
        for p in m.parameters():
            p.grad = None
        loss_step = step(x)
        loss_step.backward()
        assert len(bwd_pre_seen) == 1
        assert len(bwd_seen) == 1
        grads_step = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

        for p in m.parameters():
            p.grad = None
        loss_eager = loss_fn(m(x))
        loss_eager.backward()
        grads_eager = [p.grad.numpy().copy() for p in m.parameters() if p.grad is not None]

        for gs, ge in zip(grads_step, grads_eager):
            assert np.allclose(gs, ge, rtol=1e-4, atol=1e-5)
    finally:
        h_bpre.remove()
        h_bfull.remove()

    # After removing backward hooks, compiled step cache hits resume.
    hits_before = next(iter(step.cache.values())).n_hits
    step(x).backward()
    assert next(iter(step.cache.values())).n_hits == hits_before + 1


@pytest.mark.usefixtures("inference")
def test_global_module_hook_and_internal_weight_norm_hook() -> None:
    """Global hooks route to eager while active; internal weight_norm pre-hooks still compile."""
    # Internal weight_norm hook is traced into the graph, not treated as an eager-only user hook.
    wn_model = nn.Sequential(nn.utils.weight_norm(nn.Linear(8, 8)), nn.ReLU()).to(COMPILE_DEVICE)
    wn_model.eval()
    assert not model_has_active_hooks(wn_model)
    cm = lucid.compile(wn_model)
    x = metal_tensor(4, 8)
    got = cm(x).numpy()
    want = wn_model(x).numpy()
    assert cm.cache_info()["entries"] == 1
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)

    # Global forward hook routes to eager while registered, then resumes compiled cache.
    calls: list[str] = []
    gh = nn.hooks.register_module_forward_hook(
        lambda mod, inp, out: calls.append(type(mod).__name__)
    )
    try:
        assert model_has_active_hooks(wn_model)
        _ = cm(x)
        assert len(calls) > 0
    finally:
        gh.remove()
    assert not model_has_active_hooks(wn_model)


def test_parameter_and_mid_forward_tensor_hooks_under_compile_and_steps() -> None:
    """Parameter tensor hooks and mid-forward tensor hooks work across compile entrypoints."""
    m = _DropScaleNet(p=0.0).to(COMPILE_DEVICE)
    m.eval()
    x = metal_tensor(4, 8)

    opt = lucid.optim.SGD(m.parameters(), lr=0.1)
    fstep = lucid.compile.fused_step(m, lambda y: (y * y).sum(), opt)
    _ = fstep(x)
    assert fstep.cache_info()["entries"] == 1

    # Registering a tensor hook on a parameter routes fused_step to eager so the hook runs,
    # and removing it resumes compiled execution.
    seen_param_grads: list[int] = []
    hp = m.fc1.weight.register_hook(lambda g: (seen_param_grads.append(g.numel()), g * 0.0)[1])
    try:
        assert model_has_active_hooks(m)
        w_before = m.fc1.weight.detach().cpu().tolist()
        _ = fstep(x)
        assert len(seen_param_grads) == 1
        assert m.fc1.weight.detach().cpu().tolist() == w_before
    finally:
        hp.remove()
    assert not model_has_active_hooks(m)

    # Segmented make_step respects module hooks and mid-forward tensor hooks.
    seq = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).to(COMPILE_DEVICE)
    seg_step = lucid.compile.make_step(seq, lambda y: (y * y).sum(), segments=2)
    seg_step(x).backward()
    assert all(len(c) == 1 for c in seg_step.cache)

    mod_hook_calls: list[int] = []
    hm = seq[0].register_forward_hook(lambda mod, inp, out: mod_hook_calls.append(out.numel()))
    try:
        for p in seq.parameters():
            p.grad = None
        seg_step(x).backward()
        assert len(mod_hook_calls) == 1
    finally:
        hm.remove()

    # Mid-forward tensor hook inside forward falls back cleanly to eager under
    # CompiledModule, make_step, segmented make_step, and fused_step.
    class _MidHookLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)
            self.hook_calls = 0

        def forward(self, inp: lucid.Tensor) -> lucid.Tensor:
            h = self.fc(inp)
            if h.requires_grad or not lucid.is_grad_enabled():

                def _scale(g: lucid.Tensor) -> lucid.Tensor:
                    self.hook_calls += 1
                    return g * 0.25

                h.register_hook(_scale)
            return h * 2.0

    mh = _MidHookLinear().to(COMPILE_DEVICE)
    cm_mh = lucid.compile(mh)
    out_mh = cm_mh(x)
    out_mh.sum().backward()
    assert mh.hook_calls == 1

    mh.hook_calls = 0
    for p in mh.parameters():
        p.grad = None
    step_mh = lucid.compile.make_step(mh, lambda y: y.sum())
    step_mh(x).backward()
    assert mh.hook_calls == 1

    seq_mh = nn.Sequential(_MidHookLinear(), nn.ReLU()).to(COMPILE_DEVICE)
    seg_mh = lucid.compile.make_step(seq_mh, lambda y: y.sum(), segments=2)
    seg_mh(x).backward()
    assert seq_mh[0].hook_calls == 1  # type: ignore[union-attr]
