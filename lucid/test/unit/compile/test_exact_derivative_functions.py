"""An exact-derivative ``Function`` compiles when eager also runs its backward (LCD-570b).

A Lucid ``Function`` that sets :attr:`~lucid.autograd.Function._exact_derivative`
(:func:`~lucid.autograd.checkpoint`'s ``_CheckpointFn``) has a ``backward``
equal to the recorded ``forward``'s derivative.  When at least one positional
input requires grad (including non-reentrant ``checkpoint``'s anchor) or
descends from a differentiated source, eager runs ``backward`` and a compiled
training step may differentiate the recorded ``forward`` directly — staying
compiled on Metal with gradients equal to eager.

When none of its positional inputs requires grad or descends from a source and
only a tensor ``forward`` closed over does — ``checkpoint(fn, x,
use_reentrant=True)`` on an ungrad ``x`` where ``fn`` closes over parameters —
eager does not run ``backward`` at all (leaving those parameters with ``None``
gradients), whereas a compiled backward would differentiate the closed-over
parameters.  That call is refused with a closure-only
:class:`~lucid.compile._core.custom_backward.CustomBackwardInTrace`.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._tensor.tensor import Tensor
from lucid.autograd import Function
from lucid.autograd.checkpoint import checkpoint
from lucid.compile import fused_step, make_step
from lucid.compile._core.custom_backward import CustomBackwardInTrace
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._entry.function import compiled_step
from lucid.test.unit.compile.test_function_compile_classes import EXACT, _scan_lucid_functions


class _CheckpointedBlock(nn.Module):
    def __init__(self, *, use_reentrant: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self._use_reentrant = use_reentrant

    def _body(self, h: Tensor) -> Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(h))))

    def forward(self, h: Tensor) -> Tensor:
        out = checkpoint(self._body, h, use_reentrant=self._use_reentrant)
        assert isinstance(out, Tensor)
        return out


def _interior_checkpoint_model(*, use_reentrant: bool, device: str) -> nn.Sequential:
    """Stem ``Linear`` -> checkpointed block -> head ``Linear``."""
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(4, 4), _CheckpointedBlock(use_reentrant=use_reentrant), nn.Linear(4, 3)
    ).to(device)


def _first_layer_checkpoint_model(*, use_reentrant: bool, device: str) -> nn.Sequential:
    """Checkpointed block fed raw ``x`` (no grad) -> head ``Linear``."""
    lucid.manual_seed(0)
    return nn.Sequential(_CheckpointedBlock(use_reentrant=use_reentrant), nn.Linear(4, 3)).to(
        device
    )


def _data(device: str) -> tuple[Tensor, Tensor]:
    lucid.manual_seed(1)
    return lucid.randn(5, 4, device=device), lucid.randn(5, 3, device=device)


def _grads(model: nn.Module) -> dict[str, list[object] | None]:
    return {n: None if p.grad is None else p.grad.tolist() for n, p in model.named_parameters()}


def _params(model: nn.Module) -> dict[str, list[object]]:
    return {n: p.tolist() for n, p in model.named_parameters()}


def _assert_close_maps(
    got: dict[str, list[object] | None], want: dict[str, list[object] | None]
) -> None:
    assert got.keys() == want.keys()
    for name in want:
        g = got[name]
        w = want[name]
        if g is None or w is None:
            assert g is w, f"{name}: got {g!r}, want {w!r}"
        else:
            ga = lucid.tensor(g).reshape(-1)
            wa = lucid.tensor(w).reshape(-1)
            assert bool(lucid.allclose(ga, wa, rtol=1e-4, atol=1e-5)), name


def _eager_grads_and_updated(
    factory: Callable[[], nn.Module], x: Tensor, t: Tensor
) -> tuple[dict[str, list[object] | None], dict[str, list[object]]]:
    model = factory()
    F.mse_loss(model(x), t).backward()
    grads = _grads(model)
    optim.SGD(model.parameters(), lr=0.25).step()
    return grads, _params(model)


def test_ast_exact_derivative_set_matches_table() -> None:
    _, _, exact_derivatives = _scan_lucid_functions()
    assert exact_derivatives == EXACT == frozenset({"_CheckpointFn"})


@pytest.mark.parametrize(
    ("kind", "use_reentrant"), [("interior", False), ("interior", True), ("first_layer", False)]
)
def test_tracked_checkpoint_compiles_under_make_step_and_matches_eager(
    device_gpu_only: str, kind: str, use_reentrant: bool
) -> None:
    device = device_gpu_only
    x, t = _data(device)
    factory = (
        (lambda: _interior_checkpoint_model(use_reentrant=use_reentrant, device=device))
        if kind == "interior"
        else (lambda: _first_layer_checkpoint_model(use_reentrant=use_reentrant, device=device))
    )
    want_grads, want_params = _eager_grads_and_updated(factory, x, t)

    model = factory()
    step = make_step(model, F.mse_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    optim.SGD(model.parameters(), lr=0.25).step()

    assert len(step.cache) == 1
    assert not step.eager_only
    _assert_close_maps(_grads(model), want_grads)
    _assert_close_maps(_params(model), want_params)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_interior_checkpoint_compiles_under_segmented_step_and_matches_eager(
    device_gpu_only: str, use_reentrant: bool
) -> None:
    device = device_gpu_only
    x, t = _data(device)
    factory = lambda: _interior_checkpoint_model(use_reentrant=use_reentrant, device=device)
    want_grads, _ = _eager_grads_and_updated(factory, x, t)

    model = factory()
    step = make_step(model, F.mse_loss, segments=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()

    assert not step.eager_only["flag"]
    _assert_close_maps(_grads(model), want_grads)


@pytest.mark.parametrize(
    ("kind", "use_reentrant"), [("interior", False), ("interior", True), ("first_layer", False)]
)
def test_tracked_checkpoint_compiles_under_fused_step_and_matches_eager(
    device_gpu_only: str, kind: str, use_reentrant: bool
) -> None:
    device = device_gpu_only
    x, t = _data(device)
    factory = (
        (lambda: _interior_checkpoint_model(use_reentrant=use_reentrant, device=device))
        if kind == "interior"
        else (lambda: _first_layer_checkpoint_model(use_reentrant=use_reentrant, device=device))
    )
    _, want_params = _eager_grads_and_updated(factory, x, t)

    model = factory()
    step = fused_step(model, F.mse_loss, optim.SGD(model.parameters(), lr=0.25))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t)

    info = step.cache_info()
    assert info["entries"] == 1
    assert not info["eager_only"]
    _assert_close_maps(_params(model), want_params)


@pytest.mark.parametrize(
    ("kind", "use_reentrant"), [("interior", False), ("interior", True), ("first_layer", False)]
)
def test_tracked_checkpoint_compiles_under_compiled_step_and_matches_eager(
    device_gpu_only: str, kind: str, use_reentrant: bool
) -> None:
    device = device_gpu_only
    x, t = _data(device)
    factory = (
        (lambda: _interior_checkpoint_model(use_reentrant=use_reentrant, device=device))
        if kind == "interior"
        else (lambda: _first_layer_checkpoint_model(use_reentrant=use_reentrant, device=device))
    )
    want_grads, _ = _eager_grads_and_updated(factory, x, t)

    model = factory()
    compiled_step(model, x, lambda y: F.mse_loss(y, t))
    _assert_close_maps(_grads(model), want_grads)


def test_reentrant_checkpoint_with_closure_only_params_is_refused(device_gpu_only: str) -> None:
    """``checkpoint(fn, x, use_reentrant=True)`` on ungrad ``x``: eager drops the block's grads."""
    device = device_gpu_only
    x, t = _data(device)
    factory = lambda: _first_layer_checkpoint_model(use_reentrant=True, device=device)
    want_grads, want_params = _eager_grads_and_updated(factory, x, t)
    # In eager, the reentrant checkpointed first block receives no grad-requiring
    # input, so its parameters stay at ``grad is None``.
    assert want_grads["0.fc1.weight"] is None
    assert want_grads["0.fc2.weight"] is None
    assert want_grads["1.weight"] is not None

    model = factory()
    with pytest.raises(
        CustomBackwardInTrace, match="receives no input that requires grad.*closed over"
    ):
        compiled_step(model, x, lambda y: F.mse_loss(y, t))

    model = factory()
    step = make_step(model, F.mse_loss)
    _NOTICED.discard("custom-backward")
    with pytest.warns(CompileFallbackWarning, match="_CheckpointFn"):
        step(x, t).backward()
    optim.SGD(model.parameters(), lr=0.25).step()
    assert len(step.eager_only) == 1
    _assert_close_maps(_grads(model), want_grads)
    _assert_close_maps(_params(model), want_params)


class _DetachedStemCheckpointedModel(nn.Module):
    """Stem output is detached before entering ``checkpoint(..., use_reentrant=True)``."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block = _CheckpointedBlock(use_reentrant=True)
        self.head = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x).detach()
        return self.head(self.block(h))


def test_reentrant_checkpoint_after_detached_parameter_path_is_refused(
    device_gpu_only: str,
) -> None:
    """A ``detach()`` cuts the gradient from ``stem``, so ``block``'s input needs no grad."""
    device = device_gpu_only
    x, t = _data(device)

    def factory() -> _DetachedStemCheckpointedModel:
        lucid.manual_seed(0)
        return _DetachedStemCheckpointedModel().to(device)

    want_grads, want_params = _eager_grads_and_updated(factory, x, t)
    assert want_grads["stem.weight"] is None
    assert want_grads["block.fc1.weight"] is None
    assert want_grads["block.fc2.weight"] is None
    assert want_grads["head.weight"] is not None

    model = factory()
    with pytest.raises(
        CustomBackwardInTrace, match="receives no input that requires grad.*closed over"
    ):
        compiled_step(model, x, lambda y: F.mse_loss(y, t))

    model = factory()
    step = make_step(model, F.mse_loss)
    _NOTICED.discard("custom-backward")
    with pytest.warns(CompileFallbackWarning, match="_CheckpointFn"):
        step(x, t).backward()
    optim.SGD(model.parameters(), lr=0.25).step()
    assert len(step.eager_only) == 1
    _assert_close_maps(_grads(model), want_grads)
    _assert_close_maps(_params(model), want_params)


class _UntrustedExactClaim(Function):
    """A user class claiming ``_exact_derivative = True`` is still refused."""

    _exact_derivative = True

    @staticmethod
    def forward(ctx: object, h: Tensor) -> Tensor:
        del ctx
        return h * 1.0

    @staticmethod
    def backward(ctx: object, g: Tensor) -> Tensor:
        del ctx
        return g * 7.0


def test_untrusted_exact_derivative_claim_is_refused(device_gpu_only: str) -> None:
    device = device_gpu_only
    x, t = _data(device)
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    with pytest.raises(
        CustomBackwardInTrace,
        match="autograd.Function _UntrustedExactClaim defines its own backward",
    ):
        compiled_step(model, x, lambda y: F.mse_loss(_UntrustedExactClaim.apply(y), t))
