"""Recurrence guards for routing all MPSGraph pad ops through ``pad_or_identity``.

Direct all-zero ``[g padTensor:...]`` calls can abort MPSGraph ANEC padding
conversion on macOS 26 (LCD-223 / #297).  Every compiled pad op in
``lucid/_C/compile/`` must route through ``pad_or_identity`` in
``OpEmitters/_AttrHelpers.h``, which returns ``x`` unchanged when all padding
amounts are zero.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_compiles,
    metal_tensor,
)

_COMPILE_DIR = Path(lucid.__file__).resolve().parent / "_C" / "compile"
_HELPER_REL = Path("OpEmitters/_AttrHelpers.h")
_CALLER_RELS = (
    Path("OpEmitters/shape/Reshape.mm"),
    Path("VjpEmitters/linalg/Products.mm"),
    Path("VjpEmitters/nn/Conv.mm"),
    Path("VjpEmitters/nn/Pool.mm"),
)


def test_pad_tensor_only_in_pad_or_identity_helper() -> None:
    """No file in ``lucid/_C/compile/`` may call ``padTensor`` outside ``_AttrHelpers.h``."""
    pad_tensor_sites: list[tuple[Path, int, str]] = []
    for path in sorted(_COMPILE_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".mm"}:
            continue
        rel = path.relative_to(_COMPILE_DIR)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "padTensor" in line:
                pad_tensor_sites.append((rel, lineno, line.strip()))

    assert len(pad_tensor_sites) == 1, (
        f"Expected exactly 1 'padTensor' occurrence in {_HELPER_REL}, "
        f"found {len(pad_tensor_sites)}: {pad_tensor_sites}"
    )
    assert pad_tensor_sites[0][0] == _HELPER_REL

    for rel in _CALLER_RELS:
        text = (_COMPILE_DIR / rel).read_text(encoding="utf-8")
        assert "pad_or_identity(" in text, f"{rel} must route padding through pad_or_identity"


class _FnModule(nn.Module):
    def __init__(self, shape: tuple[int, ...], fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        lucid.manual_seed(0)
        self.w = nn.Parameter((lucid.rand(*shape) + 0.5).to(COMPILE_DEVICE))
        self._fn = fn
        self._probes: dict[tuple[int, ...], lucid.Tensor] = {}

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = self._fn(x * self.w)
        y_shape = tuple(y.shape)
        probe = self._probes.get(y_shape)
        if probe is None:
            with lucid.no_grad():
                probe = lucid.arange(
                    1,
                    y.numel() + 1,
                    dtype=y.dtype,
                    device=COMPILE_DEVICE,
                ).reshape(y_shape)
            self._probes[y_shape] = probe
        return (y * probe).sum()


def _identity_loss(out: lucid.Tensor) -> lucid.Tensor:
    return out


def _assert_manual_vjp_grad_parity(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    fn: Callable[[lucid.Tensor], lucid.Tensor],
    atol: float = 1e-4,
) -> None:
    from lucid._C import engine as _C_engine

    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")
    _C_engine.compile.session_cache_clear()

    lucid.manual_seed(1)
    x1 = metal_tensor(*shape)
    x2 = metal_tensor(*shape)

    model = _FnModule(shape, fn)
    model.w.grad = None
    model(x2).backward()
    assert model.w.grad is not None
    want = model.w.grad.numpy().copy()

    step = lucid.compile.make_step(model, _identity_loss)
    model.w.grad = None
    step(x1).backward()
    model.w.grad = None
    step(x2).backward()

    assert not step.eager_only.snapshot(), "compiled step fell back to eager"
    assert model.w.grad is not None
    got = model.w.grad.numpy()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=atol)


class _ZeroPadForward(nn.Module):
    def __init__(self, pads: tuple[int, ...]) -> None:
        super().__init__()
        self._pads = pads

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return F.pad(x * 2.0, self._pads) + 1.0


@pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("pads", [(0, 0), (0, 0, 0, 0), (1, 0, 0, 2)])
def test_pad_emitter_zero_and_nonzero_compiled_replay(
    monkeypatch: pytest.MonkeyPatch, pads: tuple[int, ...]
) -> None:
    """``PadEmitter`` in ``Reshape.mm`` compiles and matches eager on all-zero pads."""
    x = metal_tensor(2, 3, 4)
    assert_compiles(_ZeroPadForward(pads), x)
    _assert_manual_vjp_grad_parity(monkeypatch, (2, 3, 4), lambda t: F.pad(t, pads))


@pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize(
    ("groups", "padding"),
    [(4, 0), (4, 1), (1, 0)],
)
def test_conv2d_vjp_zero_padding_compiled_replay(
    monkeypatch: pytest.MonkeyPatch, groups: int, padding: int
) -> None:
    """``Conv2dVjp`` in ``Conv.mm`` (including depthwise pre-pad) compiles with ``padding=0``."""
    from lucid._C import engine as _C_engine

    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")
    _C_engine.compile.session_cache_clear()

    lucid.manual_seed(2)
    conv = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=padding, groups=groups).to(
        COMPILE_DEVICE
    )
    x1 = metal_tensor(2, 4, 6, 6)
    x2 = metal_tensor(2, 4, 6, 6)

    def loss_fn(out: lucid.Tensor) -> lucid.Tensor:
        return (out * 1.5).sum()

    conv.zero_grad()
    loss_fn(conv(x2)).backward()
    assert conv.weight.grad is not None
    want_dw = conv.weight.grad.numpy().copy()

    step = lucid.compile.make_step(conv, loss_fn)
    conv.zero_grad()
    step(x1).backward()
    conv.zero_grad()
    step(x2).backward()

    assert not step.eager_only.snapshot(), "conv2d step fell back to eager"
    assert conv.weight.grad is not None
    np.testing.assert_allclose(conv.weight.grad.numpy(), want_dw, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (3, 5),
        (5, 3),
    ],
)
def test_products_diagonal_and_matmul_vjp_zero_pad_compiled_replay(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int]
) -> None:
    """``DiagonalVjp`` in ``Products.mm`` (zero-pad when ``span == len``) and matmul/bmm backward."""
    _assert_manual_vjp_grad_parity(monkeypatch, shape, lambda t: lucid.diagonal(t))
    rhs = metal_tensor(shape[1], 3)
    _assert_manual_vjp_grad_parity(monkeypatch, shape, lambda t: t @ rhs)
    b_rhs = metal_tensor(2, shape[1], 3)
    _assert_manual_vjp_grad_parity(
        monkeypatch, (2, shape[0], shape[1]), lambda t: lucid.bmm(t, b_rhs)
    )


@pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize(
    ("fn", "shape"),
    [
        (lambda t: F.max_pool1d(t, kernel_size=2, stride=2, padding=0), (2, 3, 6)),
        (lambda t: F.max_pool2d(t, kernel_size=2, stride=2, padding=0), (2, 3, 4, 4)),
        (
            lambda t: F.max_pool3d(t, kernel_size=2, stride=2, padding=0),
            (1, 2, 4, 4, 4),
        ),
        (lambda t: F.avg_pool2d(t, kernel_size=2, stride=2, padding=0), (2, 3, 4, 4)),
    ],
)
def test_pool_vjp_zero_padding_compiled_replay(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[[lucid.Tensor], lucid.Tensor],
    shape: tuple[int, ...],
) -> None:
    """``PoolVjp`` in ``Pool.mm`` compiles and matches eager backward without padding."""
    _assert_manual_vjp_grad_parity(monkeypatch, shape, fn)
