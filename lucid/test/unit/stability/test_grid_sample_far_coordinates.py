"""``grid_sample`` at coordinates far outside the image, NaN and infinite ones.

A grid coordinate of ``inf`` or ``1e10`` killed the interpreter with SIGBUS
on the CPU (LCD-420): the kernel cast the floored position to ``int``,
which saturated, added one, which overflowed, and the bounds check then let
a wild address through.  Sample positions now become indices by one rule
(``resample::index_of`` on the CPU, ``gpu_index_of`` on Metal): NaN reads
as 0 and anything past the int64 range saturates, which is how the
reference reads them too.

The matrix covers every path: the 4-D kernel (bilinear and nearest, zero
and border padding), the reflection the kernel reads through a reflected
grid, and the Python composites — bicubic and 5-D (LCD-426).  The images
are not square and not a power of two on a side, so a reflection that only
agrees on such sizes shows.  Each case compares the value, the input
gradient and the grid gradient with the reference.

Every case runs in a child interpreter, as in ``test_no_process_crash.py``,
so a crash fails one case instead of the run.  The child reports the
sampled value and both gradients; the parent holds them to the reference.

Where Lucid departs from the reference on purpose, the case is held to a
*stand-in* grid instead — a finite coordinate at which the reference does
what Lucid does on the original one (see ``_DEPARTURES``).
"""

import json
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.stability

_COORDS = {
    "inf": math.inf,
    "-inf": -math.inf,
    "1e10": 1e10,
    "-1e10": -1e10,
    "nan": math.nan,
    "2^31": 2.0**31,
    "-2^31": -(2.0**31),
}
_PADDINGS = ["zeros", "border", "reflection"]

# (label, image shape, modes).  4x4 is the kernel's original matrix; 3x5
# and 3x4x5 have sides that are neither equal nor powers of two.
_IMAGES = [
    ("2d-4x4", (1, 1, 4, 4), ["bilinear", "nearest", "bicubic"]),
    ("2d-3x5", (1, 1, 3, 5), ["bilinear", "nearest", "bicubic"]),
    ("3d-3x4x5", (1, 1, 3, 4, 5), ["bilinear", "nearest"]),
]

# A case's departures from the reference, by name: the reason, and the
# coordinate that stands in for the original one on that axis when the
# reference is asked what Lucid computes.
_DEPARTURES = {
    "nan-split": (
        "reference 5-D forward/backward read NaN at different positions; we follow the forward",
        1.0,  # reads the last sample, clipped: the forward's position
    ),
    "overflow-wrap": (
        "reference signed-overflow wrap (UB); Lucid keeps a saturated "
        "position off the image like its 4-D kernel",
        1e10,  # off the image, finite: reads nothing
    ),
}


def _position(coord: float, size: int, align: bool) -> float:
    """The 5-D composite's pixel position of ``coord``, in float32."""
    with np.errstate(over="ignore", invalid="ignore"):
        c = np.float32(coord) + np.float32(1)
        if align:
            return float(c * np.float32((size - 1) / 2))
        fused = np.float32(float(c) * float(np.float32(size)) - 1.0)
        return float(fused / np.float32(2))


def _departure(
    shape: tuple[int, ...], mode: str, pad: str, align: bool, coord: float, size: int
) -> str | None:
    """Which named departure, if any, one axis of a case makes."""
    if len(shape) != 5:
        return None
    pos = _position(coord, size, align)
    degenerate = align and size == 1
    if pad == "border" and math.isnan(pos):
        return "nan-split"
    if pad == "reflection" and not degenerate and not math.isfinite(pos):
        return "nan-split"
    if pad == "zeros" and mode == "bilinear" and pos >= 2.0**63:
        return "overflow-wrap"
    return None


def _axes(shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """(label, grid axes it sets): each axis on its own, then all at once."""
    names = "xyz"[: len(shape) - 2]
    single = [(name, (i,)) for i, name in enumerate(names)]
    return single + [("all" if len(names) > 2 else "both", tuple(range(len(names))))]


def _point(shape: tuple[int, ...], axes: tuple[int, ...], coord: float) -> list[float]:
    point = [0.25, -0.5, 0.75][: len(shape) - 2]
    for a in axes:
        point[a] = coord
    return point


Case = tuple[str, tuple[int, ...], str, str, bool, list[float], list[float]]


def _kinds(
    shape: tuple[int, ...], mode: str, pad: str, align: bool, point: list[float]
) -> list[str | None]:
    """The departure each coordinate of ``point`` makes, or ``None``."""
    sizes = list(shape[2:])[::-1]  # grid order: x first
    return [
        _departure(shape, mode, pad, align, coord, size)
        for coord, size in zip(point, sizes, strict=True)
    ]


def _stand_in(
    shape: tuple[int, ...], mode: str, pad: str, align: bool, point: list[float]
) -> list[float]:
    """``point`` with every departing coordinate replaced by its stand-in."""
    kinds = _kinds(shape, mode, pad, align, point)
    return [
        coord if kind is None else _DEPARTURES[kind][1]
        for coord, kind in zip(point, kinds, strict=True)
    ]


def _cases() -> list[Case]:
    """(name, image shape, mode, padding, align_corners, point, stand-in)."""
    out = []
    for label, shape, modes in _IMAGES:
        for mode in modes:
            for pad in _PADDINGS:
                for align in (False, True):
                    corners = "corners" if align else "edges"
                    for cname, coord in _COORDS.items():
                        for axis_label, axes in _axes(shape):
                            name = (
                                f"{label}-{mode}-{pad}-{corners}-{cname}-{axis_label}"
                            )
                            point = _point(shape, axes, coord)
                            stand_in = _stand_in(shape, mode, pad, align, point)
                            out.append((name, shape, mode, pad, align, point, stand_in))
    return out


CASES = _cases()


def _params() -> list[Any]:
    return [pytest.param(*case, id=case[0]) for case in CASES]


# ``rotate`` (the engine's, behind the image transforms) cast its rounded
# source position to int the same way.  (name, angle in degrees, cy, cx).
ROTATIONS = [
    ("rotate-nan-angle", math.nan, 1.5, 1.5),
    ("rotate-inf-angle", math.inf, 1.5, 1.5),
    ("rotate-cy-1e30", 30.0, 1e30, 1.5),
    ("rotate-cy-inf", 30.0, math.inf, 1.5),
    ("rotate-cy-3e9", 30.0, 3e9, 1.5),
    ("rotate-cx--inf", 30.0, 1.5, -math.inf),
]


_CHILD = """
import json, math, sys
import numpy as np
import lucid
import lucid.nn.functional as F

D = sys.argv[1]

def listed(t):
    return [None if math.isnan(v) else v for v in np.asarray(t.to("cpu").numpy(), dtype=np.float64).ravel().tolist()]

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

for case in json.loads(sys.stdin.read()):
    name = case[0]
    print("@@BEGIN " + name, flush=True)
    if name.startswith("rotate"):
        _, angle, cy, cx = case
        image = np.arange(16.0, dtype=np.float32).reshape(1, 1, 4, 4)
        x = lucid.tensor(image + 1.0, device=D)
        y = _wrap(_C_engine.nn.rotate(_unwrap(x), angle, cy, cx))
        print("@@END " + name + " " + json.dumps({"y": listed(y)}), flush=True)
        continue
    _, shape, mode, pad, align, point = case
    image = np.arange(float(math.prod(shape)), dtype=np.float32).reshape(shape)
    grid = np.array(point, dtype=np.float32).reshape((1,) * (len(shape) - 1) + (len(point),))
    x = lucid.tensor(image, device=D, requires_grad=True)
    g = lucid.tensor(grid, device=D, requires_grad=True)
    y = F.grid_sample(x, g, mode=mode, padding_mode=pad, align_corners=align)
    y.sum().backward()
    out = {"y": listed(y), "dx": listed(x.grad), "dg": listed(g.grad)}
    print("@@END " + name + " " + json.dumps(out), flush=True)
"""


def _payload() -> list[Any]:
    grids: list[Any] = [
        [name, list(shape), mode, pad, align, point]
        for name, shape, mode, pad, align, point, _ in CASES
    ]
    return grids + [list(case) for case in ROTATIONS]


def _run(device: str) -> dict[str, Any]:
    """Every case on ``device``; a child that dies is blamed on its case and
    restarted past it."""
    results: dict[str, Any] = {}
    pending = _payload()
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    while pending:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, device],
            input=json.dumps(pending),
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        current = None
        for line in done.stdout.splitlines():
            if line.startswith("@@BEGIN "):
                current = line[len("@@BEGIN ") :]
            elif line.startswith("@@END "):
                name, _, body = line[len("@@END ") :].partition(" ")
                results[name] = json.loads(body)
                current = None
        if done.returncode == 0:
            break
        blamed = current if current is not None else pending[0][0]
        results[blamed] = {"crash": f"exit {done.returncode}: {done.stderr[-400:]}"}
        pending = [case for case in pending if case[0] not in results]
    return results


_RESULTS: dict[str, dict[str, Any]] = {}


def _result(device: str, name: str) -> dict[str, Any]:
    if device not in _RESULTS:
        _RESULTS[device] = _run(device)
    return _RESULTS[device].get(name, {"crash": "the child never reached this case"})


def _array(values: list[float | None]) -> np.ndarray:
    return np.array([math.nan if v is None else v for v in values], dtype=np.float64)


def _reference(
    ref: Any,
    shape: tuple[int, ...],
    mode: str,
    pad: str,
    align: bool,
    point: list[float],
) -> dict[str, np.ndarray]:
    image = np.arange(float(math.prod(shape)), dtype=np.float32).reshape(shape)
    grid = np.array(point, dtype=np.float32).reshape(
        (1,) * (len(shape) - 1) + (len(point),)
    )
    x = ref.tensor(image, requires_grad=True)
    g = ref.tensor(grid, requires_grad=True)
    y = ref.nn.functional.grid_sample(
        x, g, mode=mode, padding_mode=pad, align_corners=align
    )
    y.sum().backward()
    return {
        "y": y.detach().double().numpy().ravel(),
        "dx": x.grad.double().numpy().ravel(),
        "dg": g.grad.double().numpy().ravel(),
    }


def _check(
    got: dict[str, Any], want: dict[str, np.ndarray], keys: tuple[str, ...]
) -> None:
    for key in keys:
        np.testing.assert_allclose(
            _array(got[key]), want[key], atol=1e-5, equal_nan=True, err_msg=key
        )


@pytest.mark.parametrize("name,shape,mode,pad,align,point,stand_in", _params())
def test_far_coordinates(
    ref: Any,
    device: str,
    name: str,
    shape: tuple[int, ...],
    mode: str,
    pad: str,
    align: bool,
    point: list[float],
    stand_in: list[float],
) -> None:
    got = _result(device, name)
    assert (
        "crash" not in got
    ), f"{name} on {device} killed the interpreter: {got['crash']}"
    want = _reference(ref, shape, mode, pad, align, point)
    kinds = {k for k in _kinds(shape, mode, pad, align, point) if k is not None}
    if not kinds:
        _check(got, want, ("y", "dx", "dg"))
        return
    # A departure.  Under nan-split the value is still the reference's (its
    # forward); the stand-in grid makes the reference do what Lucid does,
    # gradients included.
    if kinds == {"nan-split"}:
        _check(got, want, ("y",))
    stand_in_want = _reference(ref, shape, mode, pad, align, stand_in)
    _check(got, stand_in_want, ("y", "dx", "dg"))


def test_case_names_are_unique() -> None:
    names = [case[0] for case in CASES]
    assert len(names) == len(set(names))


def test_every_departure_has_a_reason() -> None:
    for reason, stand_in in _DEPARTURES.values():
        assert reason and math.isfinite(stand_in)


@pytest.mark.parametrize(
    "name,angle,cy,cx", ROTATIONS, ids=[case[0] for case in ROTATIONS]
)
def test_rotate_far_or_nan_centre(
    device: str, name: str, angle: float, cy: float, cx: float
) -> None:
    # A far centre puts every source outside the image (zeros); a NaN
    # position reads sample 0, the one rule both devices share.
    got = _result(device, name)
    assert (
        "crash" not in got
    ), f"{name} on {device} killed the interpreter: {got['crash']}"
    y = _array(got["y"])
    far = math.isinf(cx) or abs(cy) > 1e9
    np.testing.assert_array_equal(y, np.zeros(16) if far else np.full(16, 1.0))


@pytest.mark.parametrize(
    "dtype_name,a_val,b_val,c_val,want_fma,want_two_round",
    [
        ("float16", 1001.0, 3.0, -1.0, 3002.0, 3004.0),
        ("bfloat16", 129.0, 3.0, -1.0, 386.0, 388.0),
        ("float32", 1e10, 3.0, -1.0, 29999998976.0, 30000001024.0),
        (
            "float64",
            float(2**52 + 1),
            3.0,
            -1.0,
            float(3 * 2**52 + 2),
            float(3 * 2**52 + 4),
        ),
    ],
)
def test_fma_single_rounding_and_gradients(
    device: str,
    dtype_name: str,
    a_val: float,
    b_val: float,
    c_val: float,
    want_fma: float,
    want_two_round: float,
) -> None:
    """LCD-433: _C_engine.fma rounds once where a * b + c rounds twice."""
    import lucid
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    if device == "metal" and dtype_name == "float64":
        pytest.skip("float64 is CPU-only")
    dt = getattr(lucid, dtype_name)
    a = lucid.tensor([a_val, 2.0], dtype=dt, device=device, requires_grad=True)
    b = lucid.tensor(b_val, dtype=dt, device=device, requires_grad=True)
    c = lucid.tensor([c_val, 1.0], dtype=dt, device=device, requires_grad=True)

    two_round = a * b + c
    fused = _wrap(_C_engine.fma(_unwrap(a), _unwrap(b), _unwrap(c)))
    assert float(two_round[0].item()) == want_two_round
    assert float(fused[0].item()) == want_fma

    ga, gb, gc = lucid.autograd.grad(fused.sum(), (a, b, c), create_graph=True)
    np.testing.assert_allclose(ga.to("cpu").numpy(), [b_val, b_val])
    np.testing.assert_allclose(gb.to("cpu").numpy(), a_val + 2.0)
    np.testing.assert_allclose(gc.to("cpu").numpy(), [1.0, 1.0])
    (ga.sum() + gb).backward()
    assert a.grad is not None and b.grad is not None
    np.testing.assert_allclose(a.grad.to("cpu").numpy(), [1.0, 1.0])
    np.testing.assert_allclose(b.grad.to("cpu").numpy(), 2.0)

    i = lucid.tensor([1, 2], dtype=lucid.int32, device=device)
    with pytest.raises(_C_engine.InvalidArgument, match="floating-point"):
        _C_engine.fma(_unwrap(i), _unwrap(i), _unwrap(i))


def test_fma_bf16_double_rounding_traps_and_random_corpus(device: str) -> None:
    """LCD-433: bfloat16 fma avoids f32->bf16 double rounding via round-to-odd."""
    import lucid
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    traps = [
        (17.0 / 16.0, 17.0 / 16.0, 2.0**-40, 1.1328125),
        (19.0 / 16.0, 17.0 / 16.0, -(2.0**-40), 1.2578125),
        (-17.0 / 16.0, 17.0 / 16.0, -(2.0**-40), -1.1328125),
        (-19.0 / 16.0, 17.0 / 16.0, 2.0**-40, -1.2578125),
        (17.0 / 16.0, 17.0 / 16.0, 2.0**-120, 1.1328125),
        (19.0 / 16.0, 17.0 / 16.0, -(2.0**-120), 1.2578125),
    ]
    a_t = lucid.tensor([r[0] for r in traps], dtype=lucid.bfloat16, device=device)
    b_t = lucid.tensor([r[1] for r in traps], dtype=lucid.bfloat16, device=device)
    c_t = lucid.tensor([r[2] for r in traps], dtype=lucid.bfloat16, device=device)
    got_traps = _wrap(_C_engine.fma(_unwrap(a_t), _unwrap(b_t), _unwrap(c_t)))
    assert got_traps.to("cpu").tolist() == [r[3] for r in traps]

    rng = np.random.default_rng(433)
    # 2,000 general finite bf16 triples + 2,000 halfway-product trap triples.
    exps = rng.integers(-60, 60, size=(3, 2000))
    mants = rng.integers(128, 256, size=(3, 2000)) / 128.0
    signs = rng.choice([-1.0, 1.0], size=(3, 2000))
    gen = signs * mants * (2.0**exps)

    odd_a = (1.0 + (2 * rng.integers(0, 8, size=2000) + 1) / 16.0) * rng.choice(
        [-1.0, 1.0], size=2000
    )
    odd_b = (1.0 + (2 * rng.integers(0, 8, size=2000) + 1) / 16.0) * rng.choice(
        [-1.0, 1.0], size=2000
    )
    tiny_c = rng.choice([-1.0, 1.0], size=2000) * (
        2.0 ** rng.integers(-120, -10, size=2000)
    )

    a_all = np.concatenate([gen[0], odd_a])
    b_all = np.concatenate([gen[1], odd_b])
    c_all = np.concatenate([gen[2], tiny_c])

    la = lucid.tensor(a_all.tolist(), dtype=lucid.bfloat16, device=device)
    lb = lucid.tensor(b_all.tolist(), dtype=lucid.bfloat16, device=device)
    lc = lucid.tensor(c_all.tolist(), dtype=lucid.bfloat16, device=device)
    got = _wrap(_C_engine.fma(_unwrap(la), _unwrap(lb), _unwrap(lc))).to("cpu").tolist()

    from fractions import Fraction

    qa = la.to("cpu").tolist()
    qb = lb.to("cpu").tolist()
    qc = lc.to("cpu").tolist()
    rto_f32: list[float] = []
    for x, y, z in zip(qa, qb, qc, strict=True):
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        s = float(np.float32(float(exact)))
        diff = exact - Fraction(s)
        if diff != 0:
            s = float(
                np.nextafter(
                    np.float32(s),
                    np.float32(math.inf if diff > 0 else -math.inf),
                )
            )
        rto_f32.append(s)
    want = lucid.tensor(rto_f32, dtype=lucid.bfloat16).tolist()
    assert got == want
