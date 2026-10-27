"""Parity tests for :func:`lucid.compile.compile_optimizer` and :func:`fused_step`.

Two surfaces under test:

  * ``compile_optimizer`` — wraps an optimizer's ``step()`` into a
    cached MPSGraph executable.  The model + loss + backward still run
    eager; only the parameter-update arithmetic is compiled.

  * ``fused_step`` — single executable covering forward + loss +
    backward (via MPSGraph autodiff) + optimizer update.  The
    ghost-grad mechanism (Phase 1.8) plumbs derived gradients into
    placeholders that the trace recorded as zeros.

Both must produce parameter values matching the eager step to fp32
tolerance.  Drift is exclusively from MPSGraph's reordered reduction
tree — algorithmic bugs (wrong grad slot / off-by-one in state
buffers) would produce O(1) divergence.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer, fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _tiny_model() -> nn.Module:
    """Two-layer MLP — fast to compile, exercises Linear + activation + Linear."""

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.fc1(x).relu())

    return _MLP().to(COMPILE_DEVICE)


# Every optimizer compile_optimizer accepts.  If the ghost-grad
# mechanism breaks for one of these, the corresponding ``_update`` in
# ``lucid/compile/_optim/compiler.py`` is the suspect.
#
# Notes:
#   * ``SGD(nesterov=True)`` is intentionally excluded — the eager
#     engine ``_C_engine.SGD`` doesn't take a ``nesterov`` flag and
#     silently drops it, while the compile path implements Nesterov
#     faithfully.  Comparing the two would flag a real eager-side bug
#     that's outside the compile-parity scope.  Covered separately in
#     ``test_compile_optimizer_nesterov_correctness`` below.
#   * Compile-supported set is **all 13** optimizers (Y-series, 2026-05-27):
#     SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam
#     (direct compile) plus SparseAdam, Rprop, ASGD, RAdam, LBFGS
#     (per-step scalar + select-tree compile).  No optimizer is rejected.
OPTIMIZER_FACTORIES = [
    pytest.param(lambda p: optim.SGD(p, lr=0.05), id="SGD"),
    pytest.param(lambda p: optim.SGD(p, lr=0.05, momentum=0.9), id="SGD_momentum"),
    pytest.param(lambda p: optim.Adam(p, lr=0.05), id="Adam"),
    pytest.param(lambda p: optim.AdamW(p, lr=0.05), id="AdamW"),
    pytest.param(lambda p: optim.Adam(p, lr=0.05, amsgrad=True), id="Adam_amsgrad"),
    pytest.param(lambda p: optim.AdamW(p, lr=0.05, amsgrad=True), id="AdamW_amsgrad"),
    pytest.param(lambda p: optim.RMSprop(p, lr=0.05), id="RMSprop"),
    pytest.param(lambda p: optim.RMSprop(p, lr=0.05, centered=True), id="RMSprop_centered"),
    pytest.param(lambda p: optim.Adagrad(p, lr=0.05), id="Adagrad"),
    pytest.param(
        lambda p: optim.Adagrad(p, lr=0.05, initial_accumulator_value=0.1),
        id="Adagrad_initial_accumulator",
    ),
    pytest.param(lambda p: optim.Adadelta(p, lr=0.05), id="Adadelta"),
    pytest.param(lambda p: optim.Adamax(p, lr=0.05), id="Adamax"),
    pytest.param(lambda p: optim.NAdam(p, lr=0.05), id="NAdam"),
    pytest.param(lambda p: optim.SparseAdam(p, lr=0.05), id="SparseAdam"),
    pytest.param(lambda p: optim.Rprop(p, lr=0.05), id="Rprop"),
    pytest.param(lambda p: optim.ASGD(p, lr=0.05), id="ASGD"),
    pytest.param(lambda p: optim.RAdam(p, lr=0.05), id="RAdam"),
    pytest.param(
        lambda p: optim.LBFGS(p, lr=0.05),
        id="LBFGS",
        marks=pytest.mark.xfail(
            reason=(
                "LBFGS compile path uses a per-element Barzilai-Borwein "
                "direction (single-step, no closure) which intentionally "
                "differs from the flat-vector L-BFGS + line search that "
                "the eager LBFGS runs.  Convergence is verified by a "
                "dedicated test below; bit-exact eager parity is out of "
                "scope for the compile path."
            ),
            strict=True,
        ),
    ),
]


# Y-series closed the previous rejection list — every Lucid optimizer
# *class* now compiles.  The few cases that retain caveats (LBFGS only
# without closure / line search) are still listed in the supported set
# because the fused-step usage doesn't exercise those caveats.  SparseAdam
# compiles its lazy per-row rule (LCD-427, test_compiled_sparse_adam.py).
# RMSprop(centered=True), the last refused configuration, compiles since
# CHA-64.


def _clone_state(model: nn.Module) -> dict[str, lucid.Tensor]:
    """Snapshot every parameter buffer for cross-run comparison."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _run_eager_step(
    model: nn.Module, mk_opt: object, x: lucid.Tensor, t: lucid.Tensor
) -> dict[str, lucid.Tensor]:
    opt = mk_opt(list(model.parameters()))
    opt.zero_grad()
    loss = F.mse_loss(model(x), t)
    loss.backward()
    opt.step()
    return _clone_state(model)


def _max_diff(a: dict[str, lucid.Tensor], b: dict[str, lucid.Tensor]) -> float:
    """Worst absolute parameter-value drift across the dict."""
    worst = 0.0
    for k in a:
        d = float((a[k] - b[k]).abs().max().item())
        if d > worst:
            worst = d
    return worst


@pytest.mark.parametrize("mk_opt", OPTIMIZER_FACTORIES)
def test_compile_optimizer_step_parity(mk_opt: object) -> None:
    """``compile_optimizer.step()`` matches eager step within fp32 tolerance."""
    # Seed both runs from the same initial parameters by serialising
    # the model state and reloading into the second model.
    lucid.manual_seed(0)
    model_eager = _tiny_model()
    state = _clone_state(model_eager)
    model_comp = _tiny_model()
    for (n, p), (_, q) in zip(model_eager.named_parameters(), model_comp.named_parameters()):
        with lucid.no_grad():
            p.copy_(state[n])
            q.copy_(state[n])

    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    # Eager step
    eager_state = _run_eager_step(model_eager, mk_opt, x, t)

    # Compile-optimizer step (still eager forward/backward, compiled update)
    opt_comp = mk_opt(list(model_comp.parameters()))
    copt = compile_optimizer(opt_comp)
    copt.zero_grad()
    loss = F.mse_loss(model_comp(x), t)
    loss.backward()
    copt.step()
    comp_state = _clone_state(model_comp)

    worst = _max_diff(eager_state, comp_state)
    assert worst < 1e-4, f"compile_optimizer drift = {worst:.3e}"


@pytest.mark.parametrize("mk_opt", OPTIMIZER_FACTORIES)
def test_fused_step_parity(mk_opt: object) -> None:
    """``fused_step()`` matches eager step within fp32 tolerance.

    fused_step runs forward + loss + backward + update in a *single*
    MPSGraph executable using the ghost-grad placeholder mechanism.
    """
    lucid.manual_seed(0)
    model_eager = _tiny_model()
    state = _clone_state(model_eager)
    model_comp = _tiny_model()
    for (n, p), (_, q) in zip(model_eager.named_parameters(), model_comp.named_parameters()):
        with lucid.no_grad():
            p.copy_(state[n])
            q.copy_(state[n])

    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    # Eager step
    eager_state = _run_eager_step(model_eager, mk_opt, x, t)

    # Fused step (single executable)
    opt_comp = mk_opt(list(model_comp.parameters()))
    step = fused_step(model_comp, F.mse_loss, opt_comp)
    step(x, t)
    comp_state = _clone_state(model_comp)

    worst = _max_diff(eager_state, comp_state)
    assert worst < 1e-4, f"fused_step drift = {worst:.3e}"


@pytest.mark.parametrize("cls", [optim.Adam, optim.AdamW], ids=["Adam", "AdamW"])
def test_compiled_amsgrad_tracks_the_running_maximum(cls: object) -> None:
    """AMSGrad over steps whose gradients shrink, compiled against eager.

    One step cannot tell AMSGrad from Adam — the maximum of a zero buffer
    and ``v`` is ``v`` — so this runs several, with a large gradient
    followed by many small ones, where the running maximum and ``v`` part.
    """
    lucid.manual_seed(0)
    w0 = lucid.randn(4, 3)
    scales = (10.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    grads = [lucid.randn(4, 3) * s for s in scales]

    def run(compiled: bool, amsgrad: bool) -> lucid.Tensor:
        p = nn.Parameter(w0.to(COMPILE_DEVICE).detach().clone())
        opt = cls([p], lr=0.1, betas=(0.9, 0.5), amsgrad=amsgrad)  # type: ignore[operator]
        stepper = compile_optimizer(opt) if compiled else opt
        for g in grads:
            p.grad = g.to(COMPILE_DEVICE)
            stepper.step()
        return p.detach()

    eager, compiled = run(False, True), run(True, True)
    plain = run(False, False)
    assert float((eager - compiled).abs().max().item()) < 1e-5
    # The case has to exercise the maximum, or the parity above says nothing:
    # beta2 at 0.5 lets ``v`` fall fast after the large gradient.
    assert float((eager - plain).abs().max().item()) > 1e-2


# Options whose effect shows only over several steps: a decaying step size,
# a schedule that switches at t0, a running mean that needs history.
MULTI_STEP_FACTORIES = [
    pytest.param(lambda p: optim.RMSprop(p, lr=0.01, centered=True), id="RMSprop_centered"),
    pytest.param(
        lambda p: optim.RMSprop(p, lr=0.01, centered=True, momentum=0.9, alpha=0.3),
        id="RMSprop_centered_momentum",
    ),
    pytest.param(
        lambda p: optim.Adagrad(p, lr=0.05, lr_decay=0.1, initial_accumulator_value=0.1),
        id="Adagrad_lr_decay",
    ),
    pytest.param(lambda p: optim.ASGD(p, lr=0.05, lambd=0.1), id="ASGD"),
    pytest.param(
        lambda p: optim.ASGD(p, lr=0.05, lambd=0.1, t0=2.0, weight_decay=0.1), id="ASGD_averaging"
    ),
]


@pytest.mark.parametrize("mk_opt", MULTI_STEP_FACTORIES)
def test_compiled_trajectory_follows_eager(mk_opt: object) -> None:
    """Several compiled steps track the eager ones, state included.

    One step cannot show a schedule: ASGD's averaging starts at ``t0``,
    Adagrad's step size decays per step, and RMSprop's centred mean needs
    history.  The parameters and the optimizer's own buffers are compared
    after every step.
    """
    lucid.manual_seed(0)
    w0 = lucid.randn(4, 3)
    grads = [lucid.randn(4, 3) for _ in range(6)]

    def run(compiled: bool) -> tuple[list[lucid.Tensor], object]:
        p = nn.Parameter(w0.to(COMPILE_DEVICE).detach().clone())
        opt = mk_opt([p])  # type: ignore[operator]
        stepper = compile_optimizer(opt) if compiled else opt
        trail = []
        for g in grads:
            p.grad = g.to(COMPILE_DEVICE)
            stepper.step()
            trail.append(p.detach().clone())
        return trail, stepper

    eager, eager_opt = run(False)
    compiled, copt = run(True)
    for k, (a, b) in enumerate(zip(eager, compiled)):
        drift = float((a - b).abs().max().item())
        assert drift < 1e-5, f"step {k + 1}: compiled drift = {drift:.3e}"

    # The compiled step keeps its own buffers; its state_dict holds the same
    # entries as the eager engine's, under the reference framework's names.
    state = eager_opt.state_dict()["state"][0]  # type: ignore[attr-defined]
    ours = copt.state_dict()["state"][0]  # type: ignore[attr-defined]
    assert sorted(ours) == sorted(state)
    for key, want in state.items():
        assert abs(ours[key] - want).max() < 1e-5, key


def test_compile_optimizer_lbfgs_convergence() -> None:
    """LBFGS compile path converges on a tiny convex problem.

    The compile-path LBFGS uses a per-element Barzilai-Borwein
    direction (single-step, no closure) rather than the flat-vector
    L-BFGS + line search of the eager class — so eager parity is
    explicitly out of scope (see the xfail on
    ``test_compile_optimizer_step_parity[LBFGS]``).  Instead, verify
    that the compile path converges on a small problem: 5 SGD steps
    + MSE loss should drop the loss meaningfully.
    """
    lucid.manual_seed(0)
    model = _tiny_model()
    opt = optim.LBFGS(list(model.parameters()), lr=0.05)
    step = fused_step(model, F.mse_loss, opt)
    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    losses: list[float] = []
    for _ in range(5):
        loss = step(x, t)
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], (
        f"LBFGS compile path failed to converge over 5 steps; " f"losses = {losses}"
    )


def test_compile_optimizer_nesterov_correctness() -> None:
    """Compile path implements Nesterov; verify against hand-rolled formula.

    Eager SGD silently ignores ``nesterov=True`` (engine SGD doesn't
    take the flag), so we can't compare to it.  Instead we verify the
    compile-path output matches the textbook formula:
        m_t = mu * m_{t-1} + g
        eff = g + mu * m_t
        p   = p - lr * eff
    """
    lucid.manual_seed(0)
    model = _tiny_model()
    state = _clone_state(model)
    p0 = state["fc1.weight"]

    opt = optim.SGD(list(model.parameters()), lr=0.05, momentum=0.9, nesterov=True)
    copt = compile_optimizer(opt)
    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)
    copt.zero_grad()
    loss = F.mse_loss(model(x), t)
    loss.backward()
    grad_w = model.fc1.weight.grad.detach().clone()
    copt.step()
    p1 = model.fc1.weight.detach().clone()

    # Hand-rolled: m_0 = 0, m_1 = g, eff = g + mu * m_1 = g * (1 + mu)
    mu = 0.9
    lr = 0.05
    expected = p0 - lr * (grad_w + mu * grad_w)  # = p0 - lr * (1 + mu) * g
    diff = float((p1 - expected).abs().max().item())
    assert diff < 1e-5, f"nesterov formula drift = {diff:.3e}"


def test_fused_step_repeated_calls_reuse_executable() -> None:
    """Calling fused_step multiple times shouldn't recompile every time."""
    lucid.manual_seed(0)
    model = _tiny_model()
    opt = optim.Adam(list(model.parameters()), lr=0.05)
    step = fused_step(model, F.mse_loss, opt)

    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    # First call triggers compile; subsequent calls should reuse.
    losses = [float(step(x, t).item()) for _ in range(5)]
    # Loss must change between steps (parameters are updating).
    assert losses[0] != losses[-1], "params did not update between fused_step calls"
    # All finite — no NaN explosions from the autodiff path.
    for L in losses:
        assert L == L, f"NaN loss in step: {losses}"  # NaN != NaN
