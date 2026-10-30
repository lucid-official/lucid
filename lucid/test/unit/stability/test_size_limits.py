"""A size no tensor can have is refused before anything is allocated (LCD-669).

Sizes reached the allocator and the kernels unchecked, and three kinds of
request ended the interpreter instead of raising:

* **A wrapped element count.**  ``shape_numel`` multiplied the extents in a
  ``size_t``: ``zeros(2**33, 2**33)`` counted ``2^66 mod 2^64 = 0``
  elements, was built over no buffer, and ``sum()`` segfaulted.  A count
  that did not wrap could still wrap its byte size (``numel * 8``), and an
  extent formed from two others (a tiled axis, ``reshape``'s known product)
  could wrap before the count was taken.  ``shape_numel`` now holds every
  count to ``(2^63 - 1) / 16`` — the most whose size in bytes fits a signed
  64-bit integer at every dtype — and the ops that build an extent from
  others multiply it with an overflow check.
* **A count that sized nothing.**  ``histogramdd`` with a zero bin count
  allocated an empty counts tensor and added the first sample through a
  null pointer.
* **A request no machine can meet.**  macOS hands out address space for far
  more memory than it has — a 4 TiB ``posix_memalign`` succeeds — and only
  fails when the pages are touched, by killing the process:
  ``zeros(2**40)`` was SIGKILLed while zero-filling.  A buffer larger than
  physical memory (CPU) or than ``maxBufferLength`` (Metal) now raises
  ``OutOfMemory``, a ``MemoryError`` and a ``RuntimeError``.

Each case runs in a child interpreter (``_child_runner``), so a crash fails
the case instead of the test run.  Nothing here allocates more than a few
megabytes: every refusal happens before the buffer would be reserved.
"""

import pytest

from lucid.test.unit.stability._child_runner import run_child_cases

pytestmark = pytest.mark.stability

OK = "ok"
OVERFLOW = ("InvalidArgument", "overflowed")
OOM = ("OutOfMemory", "bytes")
BINS = ("InvalidArgument", "bins must be > 0")

# (name, code, expectation).  ``code`` runs in the child with ``lucid``, ``F``,
# ``D`` (the device) and ``R`` (``randn`` on ``D``) in scope; a tensor it leaves
# in ``out`` is forced, so a lazy Metal failure surfaces inside its own case.
# An expectation is ``OK``, ``(class name, message fragment)``, a list of
# those any of which will do, or a dict of either by device.

_SHAPE = "2**33, 2**33"

_FACTORIES = [
    ("zeros_2p33_squared", f"out = lucid.zeros({_SHAPE}, device=D)", OVERFLOW),
    ("zeros_2p62_by_4", "out = lucid.zeros(2**62, 4, device=D)", OVERFLOW),
    ("zeros_2p31_cubed", "out = lucid.zeros(2**31, 2**31, 2**31, device=D)", OVERFLOW),
    ("zeros_one_past_limit", "out = lucid.zeros(2**59, device=D)", OVERFLOW),
    ("ones", f"out = lucid.ones({_SHAPE}, device=D)", OVERFLOW),
    ("empty", f"out = lucid.empty({_SHAPE}, device=D)", OVERFLOW),
    ("full", f"out = lucid.full(({_SHAPE}), 1.5, device=D)", OVERFLOW),
    ("rand", f"out = lucid.rand({_SHAPE}, device=D)", OVERFLOW),
    ("randn", f"out = lucid.randn({_SHAPE}, device=D)", OVERFLOW),
    ("randint", f"out = lucid.randint(0, 9, ({_SHAPE}), device=D)", OVERFLOW),
    ("normal", f"out = lucid.normal(0.0, 1.0, size=({_SHAPE}), device=D)", OVERFLOW),
    ("eye", "out = lucid.eye(2**40, device=D)", OVERFLOW),
    ("linspace", "out = lucid.linspace(0.0, 1.0, 2**62, device=D)", OVERFLOW),
    ("logspace", "out = lucid.logspace(0.0, 1.0, 2**62, device=D)", OVERFLOW),
    # Its extent, 1 + (2^31 - 1) * 2^31 + (2^31 - 1), is 2^62 elements.
    ("empty_strided", "out = lucid.empty_strided((2**31, 2**31), (2**31, 1), device=D)", OVERFLOW),
    ("arange_count", "out = lucid.arange(0.0, 1e30, device=D)", OVERFLOW),
    ("arange_inf", "out = lucid.arange(0.0, float('inf'), device=D)", ("InvalidArgument", "range")),
    ("arange_nan", "out = lucid.arange(0.0, float('nan'), device=D)", ("InvalidArgument", "range")),
    (
        "arange_nan_step",
        "out = lucid.arange(0.0, 1.0, float('nan'), device=D)",
        ("LucidError", "NaN"),
    ),
    ("expand", "out = R(1).expand(2**40, 2**40)", OVERFLOW),
    # A zero extent makes the count 0, however large the others.
    (
        "zeros_empty_extent",
        "out = lucid.zeros(0, 2**30, 2**30, device=D)\n"
        "assert tuple(out.shape) == (0, 2**30, 2**30) and out.numel() == 0",
        OK,
    ),
]

# Each about 4 TiB or more: past any machine's memory and Metal's buffer limit.
_TOO_LARGE = [
    ("zeros_4tib", "out = lucid.zeros(2**20, 2**20, device=D)", OOM),
    ("ones_4tib", "out = lucid.ones(2**20, 2**20, device=D)", OOM),
    ("empty_4tib", "out = lucid.empty(2**20, 2**20, device=D)", OOM),
    ("full_4tib", "out = lucid.full((2**20, 2**20), 2.0, device=D)", OOM),
    ("randn_4tib", "out = lucid.randn(2**20, 2**20, device=D)", OOM),
    ("eye_4tib", "out = lucid.eye(2**20, device=D)", OOM),
    ("arange_8tib", "out = lucid.arange(2**40, device=D)", OOM),
    ("histc_8tib_of_bins", "out = lucid.histc(R(10), bins=2**40)", OOM),
    ("kron_8tib", "a = lucid.arange(2**20, device=D)\nout = lucid.kron(a, a)", OOM),
    # The largest count at the widest CPU dtype: 2^62 bytes, which fits.
    (
        "near_limit_bytes",
        "out = (lucid.zeros(2**58, dtype=lucid.complex128) if D == 'cpu'"
        " else lucid.zeros(2**29, 2**29, device=D))",
        OOM,
    ),
]

_HISTOGRAM = [
    ("histogramdd_int_zero", "out = lucid.histogramdd(R(3, 4), 0)", BINS),
    ("histogramdd_all_zero", "out = lucid.histogramdd(R(10, 2), [0, 0])", BINS),
    ("histogramdd_one_zero", "out = lucid.histogramdd(R(10, 2), [3, 0])", BINS),
    ("histogramdd_negative", "out = lucid.histogramdd(R(10, 2), [-1, 3])", BINS),
    ("histogramdd_overflow", "out = lucid.histogramdd(R(10, 2), [2**40, 2**40])", OVERFLOW),
    ("histc_zero", "out = lucid.histc(R(10), bins=0)", BINS),
    ("histc_overflow", "out = lucid.histc(R(10), bins=2**62)", OVERFLOW),
    (
        "histogramdd_still_counts",
        "c, e = lucid.histogramdd(R(10, 2), [2, 3])\n"
        "assert tuple(c.shape) == (2, 3), c.shape\n"
        "assert float(c.sum().item()) == 10.0, c\n"
        "assert [tuple(t.shape) for t in e] == [(3,), (4,)], e",
        OK,
    ),
]

# Ops that build an output extent from others.
_DERIVED_EXTENTS = [
    ("tile_extent_wraps", "out = lucid.tile(R(3), (2**62,))", OVERFLOW),
    ("tile_count", "out = lucid.tile(R(3), (2**61,))", OVERFLOW),
    ("repeat_extent_wraps", "out = lucid.repeat(R(3), 2**62, dim=0)", OVERFLOW),
    ("repeat_count", "out = lucid.repeat(R(3), 2**61, dim=0)", OVERFLOW),
    # 274177 * 67280421310721 = 2^64 + 1, which wrapped to 1: the -1 became 6
    # and the view claimed 2^64 elements over six.
    ("reshape_product_wraps", "out = R(6).reshape(274177, 67280421310721, -1)", OVERFLOW),
    ("reshape_numel_wraps", "out = R(6).reshape(274177, 67280421310721, 6)", OVERFLOW),
    # Refused as the 2^60-element grid, or at the 2^40-element step before it.
    (
        "cartesian_prod",
        "a = lucid.arange(2**20, device=D)\nout = lucid.cartesian_prod(a, a, a)",
        [OVERFLOW, OOM],
    ),
    (
        "broadcast_add",
        "a = lucid.zeros(1, device=D).expand(2**20, 1, 2**20, 1)\n"
        "b = lucid.zeros(1, device=D).expand(1, 2**20, 1, 2**20)\n"
        "out = a + b",
        [OVERFLOW, OOM],
    ),
    (
        "as_strided_reach_wraps",
        "out = lucid.as_strided(R(4), (3,), (2**62,))",
        ("LucidError", "past the end"),
    ),
    (
        "as_strided_unused_stride",
        "out = lucid.as_strided(R(4), (1,), (2**62,))",
        ("InvalidArgument", "64-bit byte range"),
    ),
    (
        "as_strided_empty_offset",
        "out = lucid.as_strided(R(4), (0,), (1,), 2**62)",
        ("InvalidArgument", "64-bit byte range"),
    ),
    (
        "interpolate_nearest",
        "out = F.interpolate(R(1, 1, 2, 2), size=(2**31 - 1, 2**31 - 1))",
        OVERFLOW,
    ),
    (
        "interpolate_bilinear",
        "out = F.interpolate(R(1, 1, 2, 2), size=(2**31 - 1, 2**31 - 1), mode='bilinear')",
        OVERFLOW,
    ),
    ("interpolate_4tib", "out = F.interpolate(R(1, 1, 2, 2), size=(2**20, 2**20))", OOM),
    # 2^16 x 2^16 kernel: its 2^32 column rows wrapped to 0 in ``int``.
    (
        "unfold_kernel_area",
        "out = F.unfold(R(1, 1, 1, 1), kernel_size=(2**16, 2**16), padding=2**15)",
        ("NotImplementedError", "2^31 - 1"),
    ),
    (
        "unfold_padding",
        "out = F.unfold(R(1, 1, 2, 2), kernel_size=2, padding=2**30)",
        ("NotImplementedError", "2^31 - 1"),
    ),
    # A kernel longer than the padded input has no window; the count was
    # truncated toward zero and came out as one.
    (
        "unfold_no_window",
        "out = F.unfold(R(1, 1, 1, 1), kernel_size=(2, 1), stride=2)",
        ("ShapeMismatch", "non-positive"),
    ),
    (
        "unfold_still_answers",
        "out = F.unfold(R(1, 2, 4, 4), kernel_size=2, stride=2)\n"
        "assert tuple(out.shape) == (1, 8, 4), out.shape",
        OK,
    ),
    (
        "fold_output",
        "out = F.fold(R(1, 1, 1), output_size=(2**30, 2**30), kernel_size=1, stride=2**30)",
        OVERFLOW,
    ),
    # Metal folds through a flat ``outH * outW`` axis, which MLX holds in
    # ``int32``; formed in ``int``, 2^40 wrapped to 0.
    (
        "fold_4tib",
        "out = F.fold(R(1, 1, 1), output_size=(2**20, 2**20), kernel_size=1, stride=2**20)",
        {"cpu": OOM, "metal": ("LucidError", "int32")},
    ),
    (
        "fold_still_answers",
        "out = F.fold(R(1, 8, 4), output_size=(4, 4), kernel_size=2, stride=2)\n"
        "assert tuple(out.shape) == (1, 2, 4, 4), out.shape",
        OK,
    ),
    # MLX counts repeats in ``int``; 2^32 + 1 was truncated to 1 and the
    # result held one element of the 2^32 + 1 its shape claimed.  On the CPU
    # the same call is a real 16 GiB request, so it is not made there.
    (
        "tile_metal_count_past_int32",
        "out = lucid.tile(R(1), (2**32 + 1,)) if D == 'metal' else None",
        {"cpu": OK, "metal": ("LucidError", "int32")},
    ),
    (
        "repeat_metal_count_past_int32",
        "out = lucid.repeat(R(1), 2**32 + 1, dim=0) if D == 'metal' else None",
        {"cpu": OK, "metal": ("LucidError", "int32")},
    ),
    # Past ``n`` without replacement there are no combinations; the loop
    # used to build the remaining columns one empty step at a time.
    (
        "combinations_r_past_n",
        "out = lucid.combinations(lucid.arange(3, device=D), r=10**7)\n"
        "assert tuple(out.shape) == (0, 10**7), out.shape",
        OK,
    ),
]

CASES = _FACTORIES + _TOO_LARGE + _HISTOGRAM + _DERIVED_EXTENTS

_CHILD = r"""
import json
import sys

import lucid
import lucid.nn.functional as F

D = sys.argv[1]
lucid.manual_seed(0)


def R(*shape):
    return lucid.randn(*shape, device=D)


for name, code in json.loads(sys.stdin.read()):
    print("@@BEGIN " + name, flush=True)
    scope = {"lucid": lucid, "F": F, "D": D, "R": R}
    try:
        exec(code, scope)
        out = scope.get("out")
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(t, lucid.Tensor) and t.numel() > 0:
                t.sum().item()
        verdict = ["ok", "", []]
    except AssertionError as e:
        verdict = ["assert", str(e)[:400], []]
    except Exception as e:
        verdict = [type(e).__name__, str(e)[:400], [c.__name__ for c in type(e).__mro__]]
    print("@@END " + name + " " + json.dumps(verdict), flush=True)
"""

_RESULTS: dict[str, dict[str, list[object]]] = {}


def _result(device: str, name: str) -> list[object]:
    if device not in _RESULTS:
        _RESULTS[device] = run_child_cases(
            _CHILD, [(n, code) for n, code, _ in CASES], device, crash_extra=([],)
        )
    return _RESULTS[device].get(name, ["missing", "the child never reached this case", []])


@pytest.mark.parametrize("name,expect", [(n, e) for n, _, e in CASES], ids=[n for n, _, _ in CASES])
def test_size_is_refused_before_it_is_allocated(device: str, name: str, expect: object) -> None:
    kind, detail, mro = _result(device, name)
    assert kind != "crash", f"{name} on {device} killed the interpreter — {detail}"
    want = expect[device] if isinstance(expect, dict) else expect
    if want == OK:
        assert kind == "ok", f"{name} on {device}: {kind}: {detail}"
        return
    accepted = want if isinstance(want, list) else [want]
    matches = [(k, n) for k, n in accepted if k == kind and n in str(detail)]
    assert matches, f"{name} on {device}: {kind}: {detail} (expected one of {accepted})"
    want_kind = matches[0][0]
    # Catchable as the builtin of its kind, and as a RuntimeError like the
    # reference's refusal.
    assert "RuntimeError" in mro, f"{name} on {device}: {kind} is not a RuntimeError"
    builtin = {"InvalidArgument": "ValueError", "OutOfMemory": "MemoryError"}.get(want_kind)
    if builtin is not None:
        assert builtin in mro, f"{name} on {device}: {kind} is not a {builtin}"


def test_case_names_are_unique() -> None:
    names = [n for n, _, _ in CASES]
    assert len(names) == len(set(names))
