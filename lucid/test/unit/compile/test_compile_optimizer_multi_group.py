"""Compile optimizer — multi-group (X3) acceptance.

Backbone + head model with distinct learning rates per group.
``compile_optimizer`` compiles every ``param_group`` into one MPSGraph
executable, each parameter reading its own group's hyper-parameters
from the live groups.  Verifies that 5 SGD steps match eager training
within 1e-4 absolute (tighter than F16 because everything is F32).
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


class _BackboneHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.backbone(x).relu())


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _BackboneHead().to(COMPILE_DEVICE)
    b = _BackboneHead().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())
    return a, b


def _build_optimizer(model: nn.Module) -> optim.SGD:
    """SGD with two groups: backbone @ lr=1e-3, head @ lr=1e-2."""
    return optim.SGD(
        [
            {"params": list(model.backbone.parameters()), "lr": 1e-3},
            {"params": list(model.head.parameters()), "lr": 1e-2},
        ],
        lr=1e-3,  # default; per-group lr overrides
    )


def _trajectory(
    model: nn.Module, opt_factory, x: lucid.Tensor, t: lucid.Tensor, steps: int
) -> list[float]:
    opt = opt_factory(model)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _trajectory_compiled(
    model: nn.Module, opt_factory, x: lucid.Tensor, t: lucid.Tensor, steps: int
) -> list[float]:
    opt = opt_factory(model)
    copt = compile_optimizer(opt)
    losses: list[float] = []
    for _ in range(steps):
        copt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        copt.step()
        losses.append(float(loss.item()))
    return losses


def test_multi_group_sgd_parity() -> None:
    """Backbone(lr=1e-3) + Head(lr=1e-2): compile matches eager within 1e-4."""
    lucid.manual_seed(0)
    x = metal_tensor(8, 16)
    t = metal_tensor(8, 4)
    eager_model, comp_model = _matched_models()

    eager = _trajectory(eager_model, _build_optimizer, x, t, steps=5)
    comp = _trajectory_compiled(comp_model, _build_optimizer, x, t, steps=5)

    assert len(eager) == len(comp) == 5
    for k in range(5):
        diff = abs(eager[k] - comp[k])
        assert diff < 1e-4, (
            f"multi-group compile drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


def test_multi_group_shares_the_parent_groups() -> None:
    """Multi-group compiles into the one compiled optimizer, over the parent's groups.

    No per-group clone: a learning rate a scheduler (or a hand edit) writes
    into the parent's group is the one the next compiled step uses.
    """
    from lucid.compile._optim.compiler import _CompiledStepBase

    lucid.manual_seed(0)
    x = metal_tensor(8, 16)
    t = metal_tensor(8, 4)
    eager_model, comp_model = _matched_models()
    eager_opt = _build_optimizer(eager_model)
    opt = _build_optimizer(comp_model)
    copt = compile_optimizer(opt)
    assert isinstance(copt, _CompiledStepBase)
    assert copt.param_groups is opt.param_groups

    for k in range(3):
        for o, m in ((eager_opt, eager_model), (copt, comp_model)):
            o.zero_grad()
            F.mse_loss(m(x), t).backward()
            o.step()
        if k == 0:
            # After the first step the head's rate drops for both, written
            # the way an LR scheduler writes it.
            for o in (eager_opt, opt):
                o.param_groups[1]["lr"] = 1e-4
                o._sync_hyperparams()
    for (_, pe), (_, pc) in zip(eager_model.named_parameters(), comp_model.named_parameters()):
        assert float((pe - pc).abs().max().item()) < 1e-6
