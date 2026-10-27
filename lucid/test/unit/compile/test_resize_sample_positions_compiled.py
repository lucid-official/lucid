"""A compiled resize reads the samples eager reads (LCD-425).

The resize emitters used MPSGraph's resize kernels, whose sample positions
are not the engine's shared rule (``backend/SamplePositions.h``, LCD-403):
compiled against eager on Metal, nearest 2155 -> 5387 read one sample
elsewhere, 37x2155 -> 91x5387 read 91, bilinear blended up to 1e-3 off and
float16 2e-3.  The emitters now build the eager Metal kernel's gathers and
blends from that rule, so the matrix below — the engine's long axes, both
corner conventions, every float dtype the Metal stream resizes — holds the
forward to the eager result exactly, and the backward to it within the
reordering of a scatter-add (eager's own scatter is not ordered either).

``bicubic`` has no engine kernel: it is per-axis gathers in Python, which a
trace records op by op, so its positions are the shared rule's already; it
is covered here to keep it that way, to the generic emitters' rounding.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

DTYPES = {
    "f32": lucid.float32,
    "f16": lucid.float16,
    "bf16": lucid.bfloat16,
}
# Backward: a sum over the outputs that read one sample, in any order.
GRAD_RTOL = {"f32": 1e-5, "f16": 2e-3, "bf16": 1.6e-2}
# bicubic's composite arithmetic runs through the generic elementwise
# emitters, which round a few ulp apart from the eager kernels (as every
# compiled elementwise chain does); its positions are the shared rule's.
BICUBIC_RTOL = {"f32": 1e-6, "f16": 4e-3, "bf16": 3.2e-2}

# (input shape, output size): the engine's long axes, an up- and a
# down-sample of a plane, and a volume.
CASES_2D = [
    ((1, 1, 1, 2155), (1, 5387)),
    ((1, 1, 37, 2155), (91, 5387)),
    ((2, 3, 40, 50), (68, 85)),
    ((2, 3, 68, 85), (40, 50)),
]
CASES_3D = [
    ((1, 2, 7, 9, 11), (13, 5, 17)),
    ((1, 1, 3, 4, 997), (5, 2, 2477)),
]
MODES_2D = [
    ("nearest", None),
    ("bilinear", False),
    ("bilinear", True),
    ("bicubic", False),
    ("bicubic", True),
]
MODES_3D = [("nearest", None), ("trilinear", False), ("trilinear", True)]


def _matrix(
    cases: list[tuple[tuple[int, ...], tuple[int, ...]]],
    modes: list[tuple[str, bool | None]],
) -> list[object]:
    out = []
    for shape, size in cases:
        for mode, align in modes:
            corner = "" if align is None else ("-corners" if align else "-centres")
            sid = "x".join(map(str, shape[2:])) + "->" + "x".join(map(str, size))
            out.append(pytest.param(shape, size, mode, align, id=f"{mode}{corner}-{sid}"))
    return out


MATRIX = _matrix(CASES_2D, MODES_2D) + _matrix(CASES_3D, MODES_3D)


def _resize(x: lucid.Tensor, size: tuple[int, ...], mode: str, align: bool | None) -> lucid.Tensor:
    if align is None:
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=align)


def _input(shape: tuple[int, ...], dtype: lucid.dtype, seed: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(*shape).to(device=COMPILE_DEVICE, dtype=dtype)


def _gap(got: lucid.Tensor, want: lucid.Tensor) -> float:
    return float((got.float() - want.float()).abs().max().item())


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(("shape", "size", "mode", "align"), MATRIX)
def test_forward_matches_eager(
    shape: tuple[int, ...],
    size: tuple[int, ...],
    mode: str,
    align: bool | None,
    dtype: str,
) -> None:
    fn = lambda t: _resize(t, size, mode, align) * 1  # noqa: E731
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile(fn)
    compiled(_input(shape, DTYPES[dtype], 1))
    x = _input(shape, DTYPES[dtype], 2)
    got = compiled(x)
    info = compiled.cache_info()  # type: ignore[attr-defined]
    assert info["entries"] == 1 and not info["eager_only"], info
    want = fn(x)
    if mode != "bicubic":
        assert _gap(got, want) == 0.0
        return
    scale = max(float(want.float().abs().max().item()), 1.0)
    assert _gap(got, want) <= BICUBIC_RTOL[dtype] * scale


class _Probe(nn.Module):
    """``resize(x * w)`` weighted by a fixed non-uniform probe."""

    def __init__(
        self,
        shape: tuple[int, ...],
        size: tuple[int, ...],
        mode: str,
        align: bool | None,
        dtype: lucid.dtype,
    ) -> None:
        super().__init__()
        lucid.manual_seed(5)
        self.w = nn.Parameter((lucid.rand(*shape) + 0.5).to(COMPILE_DEVICE, dtype))
        lucid.manual_seed(11)
        out_shape = (*shape[:2], *size)
        self.probe = (lucid.rand(*out_shape) + 0.5).to(COMPILE_DEVICE, dtype)
        self.size, self.mode, self.align = size, mode, align

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = _resize(x * self.w, self.size, self.mode, self.align)
        return (y * self.probe).float().sum()


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(("shape", "size", "mode", "align"), MATRIX)
def test_backward_matches_eager(
    shape: tuple[int, ...],
    size: tuple[int, ...],
    mode: str,
    align: bool | None,
    dtype: str,
) -> None:
    dt = DTYPES[dtype]
    model = _Probe(shape, size, mode, align, dt)
    x = _input(shape, dt, 2)
    model(x).backward()
    want = model.w.grad.detach().clone()  # type: ignore[union-attr]

    _C_engine.compile.session_cache_clear()
    step = lucid.compile.make_step(model, lambda out: out)
    model.w.grad = None
    step(_input(shape, dt, 1)).backward()
    model.w.grad = None
    step(x).backward()
    assert not len(step.eager_only), "the compiled step fell back to eager"  # type: ignore[attr-defined]
    got = model.w.grad
    assert got is not None
    scale = max(float(want.float().abs().max().item()), 1.0)
    assert _gap(got, want) <= GRAD_RTOL[dtype] * scale


def test_a_dynamic_batch_compiles_the_resize() -> None:
    """Only the trailing axes are gathered, so a symbolic batch rides through."""
    fn = lambda t: F.interpolate(t, size=(13, 5, 17), mode="trilinear") * 1  # noqa: E731
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile(fn, dynamic=True)
    for batch in (1, 3):
        x = _input((batch, 2, 7, 9, 11), lucid.float32, batch)
        assert _gap(compiled(x), fn(x)) == 0.0
    assert not compiled.cache_info()["eager_only"]  # type: ignore[attr-defined]
