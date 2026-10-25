"""A compiled step trains a custom ``Function`` as eager does.

A compiled step traces the model under ``no_grad`` and differentiates the
traced ops, so how ``apply`` hands back ``forward``'s outputs is what the
trace sees:

* a tensor ``forward`` made and marked non-differentiable was detached only
  when it needed a gradient, which under ``no_grad`` it never does, so the
  trace sent a gradient through it that eager does not — one ``fused_step``
  moved a layer eager leaves alone by 0.86.  A marked output is always
  detached now, and a marked input comes back as a detached alias, which
  the trace records as the gradient sink it is;
* an unmarked input returned as-is must stay on the trace's gradient path.
  Handing it back as a detached alias, as a first fix of LCD-543 did, cut
  the layer in front of it off (``make_step``: ``grad is None``;
  ``fused_step``: one step off by 0.43) though eager trained it.

Since LCD-565 an unverified custom ``Function`` runs eagerly and says so;
since LCD-570 ``_CheckpointFn`` (:attr:`~lucid.autograd.Function._exact_derivative`)
compiles on Metal when its inputs carry or descend from a gradient source.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.autograd import Function
from lucid.autograd.checkpoint import checkpoint
from lucid.compile import fused_step
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning


class _PassThrough(Function):
    """``h`` as-is beside ``s * 2``."""

    @staticmethod
    def forward(ctx, h, s):
        del ctx
        return h, s * 2

    @staticmethod
    def backward(ctx, grad_h, grad_s):
        del ctx
        return grad_h, grad_s * 2


def _marking(kind: str) -> type:
    """``h * 1`` beside something built from ``s``, marked non-differentiable."""

    class Marking(Function):
        @staticmethod
        def forward(ctx, h, s):
            out = s if kind == "input" else s * 2
            ctx.mark_non_differentiable(out)
            return h * 1, out

        @staticmethod
        def backward(ctx, grad_h, grad_out):
            del ctx, grad_out
            return grad_h, None

    return Marking


_BLOCKS = {
    "input returned as-is": lambda h, s: _PassThrough.apply(h, s),
    "checkpoint passing an input through": lambda h, s: checkpoint(
        lambda p, q: (p, q * 2), h, s, use_reentrant=True
    ),
    "marked input": lambda h, s: _marking("input").apply(h, s),
    "marked fresh tensor": lambda h, s: _marking("fresh").apply(h, s),
}

#: The ``Function`` each unverified block goes through, as the fallback names it.
_FUNCTIONS = {
    "input returned as-is": "_PassThrough",
    "marked input": "Marking",
    "marked fresh tensor": "Marking",
}


def _says_it_ran_eagerly(block: str):
    _NOTICED.discard("custom-backward")
    return pytest.warns(CompileFallbackWarning, match=_FUNCTIONS[block])


class _Model(nn.Module):
    def __init__(self, block: str) -> None:
        super().__init__()
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(4, 3)
        self.block = block

    def forward(self, x):
        p, q = _BLOCKS[self.block](self.a(x), self.b(x))
        return p + q


def _model(block: str, device: str) -> _Model:
    lucid.manual_seed(0)
    return _Model(block).to(device)


def _data(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return lucid.randn(5, 4).to(device), lucid.randn(5, 3).to(device)


def _eager_grads(block: str, device: str) -> dict[str, list | None]:
    model = _model(block, device)
    x, t = _data(device)
    F.mse_loss(model(x), t).backward()
    return {
        n: None if p.grad is None else p.grad.tolist()
        for n, p in model.named_parameters()
    }


def _close(a: list | None, b: list | None) -> bool:
    if a is None or b is None:
        return a is b
    flat_a = lucid.tensor(a).reshape(-1)
    flat_b = lucid.tensor(b).reshape(-1)
    return bool(lucid.allclose(flat_a, flat_b, rtol=1e-4, atol=1e-5))


@pytest.mark.parametrize("block", sorted(_BLOCKS))
def test_make_step_gradients_match_eager(block: str, device: str) -> None:
    want = _eager_grads(block, device)
    model = _model(block, device)
    x, t = _data(device)
    step = lucid.compile.make_step(model, F.mse_loss)
    if device == "metal" and block in _FUNCTIONS:
        with _says_it_ran_eagerly(block):
            step(x, t).backward()  # the first call traces
    else:
        step(x, t).backward()
        _NOTICED.discard("lowering")
    for p in model.parameters():
        p.grad = None
    step(x, t).backward()
    if device == "metal" and block not in _FUNCTIONS:
        assert len(step.cache) == 1
        assert len(step.eager_only) == 0
    else:
        assert len(step.eager_only) == 1
    got = {
        n: None if p.grad is None else p.grad.tolist()
        for n, p in model.named_parameters()
    }
    assert got.keys() == want.keys()
    for name in want:
        assert _close(got[name], want[name]), name


@pytest.mark.parametrize("block", sorted(_BLOCKS))
def test_fused_step_updates_match_eager(block: str, device_gpu_only: str) -> None:
    device = device_gpu_only
    x, t = _data(device)

    eager = _model(block, device)
    opt = optim.SGD(eager.parameters(), lr=0.5)
    F.mse_loss(eager(x), t).backward()
    opt.step()

    fused = _model(block, device)
    step = fused_step(fused, F.mse_loss, optim.SGD(fused.parameters(), lr=0.5))
    if block in _FUNCTIONS:
        with _says_it_ran_eagerly(block):
            step(x, t)
        assert step.cache_info()["eager_only"]
    else:
        step(x, t)
        assert step.cache_info()["entries"] == 1
        assert not step.cache_info()["eager_only"]

    for (name, want), got in zip(
        eager.named_parameters(), fused.parameters(), strict=True
    ):
        assert _close(got.tolist(), want.tolist()), name
