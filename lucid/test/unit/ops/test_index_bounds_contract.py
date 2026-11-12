"""Out-of-range index values: the CPU refuses them, Metal isolates them (LCD-228).

MLX's Metal gather and scatter kernels never compare an index with its
axis.  Before LCD-228 an out-of-range index read GPU memory outside the
source (``gather(arange(4), 0, [64])`` answered 7.3e28) and an
out-of-range scatter wrote outside its buffer.  Every index op now keeps
one contract:

* CPU: ``IndexError``.
* Metal (policy B, decided 2026-10-05): no host sync and no access
  outside the buffer.  A float gather answers NaN where the index is out
  of range and an integer gather answers 0.  A scatter drops the
  out-of-range update, so the base keeps its value there.  ``one_hot``
  answers a zero row.

``LUCID_STRICT_INDEX=1`` is the opt-in exception (LCD-892): Metal then
raises the CPU's IndexError, in the CPU's words, at the cost of reading the
index back.  The strict tests at the bottom pin that, and that it changes
nothing when off.

The owners are ``gpu_axis_index`` in ``lucid/_C/backend/gpu/AxisIndex.h``
and the CPU kernels' checks in ``lucid/_C/backend/cpu/IndexBounds.h``.  The
device difference is written up in ``lucid/test/audit/README.md``.

Each op is swept over the values just outside an axis of length ``N``:
``N``; ``-N - 1``, or ``-1`` for an op that never counts from the end (a
table row, a class); and ``2**40``, which a narrowing cast to int32 turns
into a valid index.  An index into an empty axis is decided from the
shapes alone, so both devices refuse it.
"""

import math
from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

N = 4  # length of the indexed axis in every case below
NAN = math.nan

_WRAPPING = {"n": N, "below": -N - 1, "huge": 2**40}
_NON_WRAPPING = {"n": N, "below": -1, "huge": 2**40}


def _ix(values: list[object], device: str) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.int64).to(device)


def _same(got: object, want: object) -> bool:
    """Equal element by element, NaN matching NaN."""
    if isinstance(got, list):
        return (
            isinstance(want, list)
            and len(got) == len(want)
            and all(_same(g, w) for g, w in zip(got, want, strict=True))
        )
    if isinstance(want, float) and math.isnan(want):
        return isinstance(got, float) and math.isnan(got)
    return got == want


def _check(device: str, run: Callable[[], lucid.Tensor], metal_answer: object) -> None:
    """CPU refuses; Metal answers ``metal_answer`` exactly."""
    if device == "cpu":
        with pytest.raises(IndexError):
            run()
        return
    got = run().tolist()
    assert _same(got, metal_answer), f"{got} != {metal_answer}"


# ── gathers: NaN (float) or 0 (integer) at the out-of-range position ─────────


def _gather_cases(bad: int) -> dict[str, tuple[Callable[[str], lucid.Tensor], object]]:
    """name -> (run on a device, Metal's answer).  Every case also reads one
    in-range position, which must come back exact."""

    def grid(device: str) -> lucid.Tensor:
        return lucid.arange(2 * N, dtype=lucid.float32).reshape(2, N).to(device)

    def table(device: str) -> lucid.Tensor:
        return lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)

    nan_row = [NAN, NAN, NAN]
    return {
        "gather": (
            lambda d: lucid.gather(grid(d), 1, _ix([[1, bad], [bad, 3]], d)),
            [[1.0, NAN], [NAN, 7.0]],
        ),
        "gather_int": (
            lambda d: lucid.gather(
                lucid.arange(N, dtype=lucid.int32).to(d) + 10, 0, _ix([2, bad], d)
            ),
            [12, 0],
        ),
        "index_select": (
            lambda d: lucid.index_select(table(d), 0, _ix([2, bad], d)),
            [[6.0, 7.0, 8.0], nan_row],
        ),
        "take": (lambda d: lucid.take(grid(d)[:, :2], _ix([3, bad], d)), [5.0, NAN]),
        "getitem": (
            lambda d: lucid.arange(N, dtype=lucid.float32).to(d)[_ix([1, bad], d)],
            [1.0, NAN],
        ),
        # Two index tensors fold into one flat index; each is checked against
        # its own dim first (``_coordinate_select``).
        "getitem_coords": (lambda d: grid(d)[_ix([1, bad], d), _ix([0, 1], d)], [4.0, NAN]),
    }


def _row_cases(bad: int) -> dict[str, tuple[Callable[[str], lucid.Tensor], object]]:
    """Ops whose index is a table row or a class: nothing lies below zero."""

    def table(device: str) -> lucid.Tensor:
        return lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)

    def logits(device: str) -> lucid.Tensor:
        return lucid.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]).to(device)

    def engine(fn: Callable[..., object], *args: object) -> lucid.Tensor:
        impls = [_unwrap(a) if isinstance(a, lucid.Tensor) else a for a in args]
        return _wrap(fn(*impls))

    ce_first = -math.log(math.exp(1) / sum(math.exp(v) for v in range(4)))
    return {
        "embedding": (
            lambda d: engine(_C_engine.nn.embedding, table(d), _ix([2, bad], d), -1),
            [[6.0, 7.0, 8.0], [NAN, NAN, NAN]],
        ),
        # Bag 0 holds row 1 and the bad index, bag 1 holds row 2.
        "embedding_bag": (
            lambda d: engine(
                _C_engine.nn.embedding_bag,
                table(d),
                _ix([1, bad, 2], d),
                lucid.tensor([0, 2], dtype=lucid.int32).to(d),
                0,
                -1,
                False,
            ),
            [[NAN, NAN, NAN], [6.0, 7.0, 8.0]],
        ),
        "nll_loss": (
            lambda d: engine(_C_engine.nn.nll_loss, logits(d), _ix([1, bad], d), None, 0),
            [-1.0, NAN],
        ),
        "cross_entropy": (
            lambda d: engine(_C_engine.nn.cross_entropy_loss, logits(d), _ix([1, bad], d), None, 0),
            [pytest.approx(ce_first, rel=1e-5), NAN],
        ),
        "one_hot": (
            lambda d: engine(_C_engine.nn.one_hot, _ix([1, bad], d), N),
            [[0, 1, 0, 0], [0, 0, 0, 0]],
        ),
    }


# No Python call site casts a user index to int32 before the engine sees it
# any more (LCD-273): ``x[i]`` and ``x[i] = v`` (API-02), the index
# composites (API-05) and the ``scatter`` / ``scatter_add`` adapters
# (API-05b) all pass it at full width, so every ``huge`` case is refused.
def _name_case_params(names: list[str], cases: list[str]) -> list[object]:
    """Every (name, case) pair."""
    return [pytest.param(name, case, id=f"{name}-{case}") for name in names for case in cases]


@pytest.mark.parametrize(
    ("name", "case"), _name_case_params(list(_gather_cases(0)), list(_WRAPPING))
)
def test_gather_family(device: str, name: str, case: str) -> None:
    run, answer = _gather_cases(_WRAPPING[case])[name]
    _check(device, lambda: run(device), answer)


@pytest.mark.parametrize("case", list(_NON_WRAPPING))
@pytest.mark.parametrize("name", list(_row_cases(0)))
def test_row_and_class_family(device: str, name: str, case: str) -> None:
    run, answer = _row_cases(_NON_WRAPPING[case])[name]
    _check(device, lambda: run(device), answer)


def test_negative_rows_never_wrap_on_metal(device_gpu_only: str) -> None:
    # -1 names the last row for gather, but no row at all for a table: it
    # must not quietly read the last embedding.
    d = device_gpu_only
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(d)
    out = _wrap(_C_engine.nn.embedding(_unwrap(table), _unwrap(_ix([-1], d)), -1))
    assert all(math.isnan(v) for v in out.tolist()[0])


# ── scatters: the out-of-range update is dropped ─────────────────────────────


def _base(device: str, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return lucid.arange(2 * N, dtype=dtype).reshape(2, N).to(device)


def _scatter_cases(bad: int) -> dict[str, tuple[Callable[[str], lucid.Tensor], object]]:
    """name -> (run, Metal's answer).  Each writes one in-range position per
    row and aims one update out of range; everything else must be the base."""
    idx = [[1, bad], [bad, 3]]

    def src(device: str, value: float, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
        return lucid.full((2, 2), value, dtype=dtype).to(device)

    def amax(d: str) -> lucid.Tensor:
        return lucid.scatter_reduce(_base(d), 1, _ix(idx, d), src(d, 100.0), "amax")

    def amin(d: str) -> lucid.Tensor:
        return lucid.scatter_reduce(_base(d), 1, _ix(idx, d), src(d, -100.0), "amin")

    def prod(d: str) -> lucid.Tensor:
        return lucid.scatter_reduce(_base(d), 1, _ix(idx, d), src(d, 3.0), "prod")

    def flat(d: str) -> lucid.Tensor:
        return lucid.zeros(N).to(d)

    def index_put(d: str) -> lucid.Tensor:
        x = flat(d)
        x.index_put_((_ix([1, bad], d),), lucid.full((2,), 10.0).to(d))
        return x

    return {
        "scatter_add": (
            lambda d: lucid.scatter_add(_base(d), 1, _ix(idx, d), src(d, 10.0)),
            [[0.0, 11.0, 2.0, 3.0], [4.0, 5.0, 6.0, 17.0]],
        ),
        "scatter_add_int64": (
            lambda d: lucid.scatter_add(
                _base(d, lucid.int64), 1, _ix(idx, d), src(d, 10, lucid.int64)
            ),
            [[0, 11, 2, 3], [4, 5, 6, 17]],
        ),
        "scatter": (
            lambda d: lucid.scatter(_base(d), 1, _ix(idx, d), src(d, 10.0)),
            [[0.0, 10.0, 2.0, 3.0], [4.0, 5.0, 6.0, 10.0]],
        ),
        "scatter_int64": (
            lambda d: lucid.scatter(_base(d, lucid.int64), 1, _ix(idx, d), src(d, 10, lucid.int64)),
            [[0, 10, 2, 3], [4, 5, 6, 10]],
        ),
        "scatter_bool": (
            lambda d: lucid.scatter(
                lucid.zeros(2, N, dtype=lucid.bool_).to(d),
                1,
                _ix(idx, d),
                lucid.ones(2, 2, dtype=lucid.bool_).to(d),
            ),
            [[False, True, False, False], [False, False, False, True]],
        ),
        "scatter_reduce_amax": (amax, [[0.0, 100.0, 2.0, 3.0], [4.0, 5.0, 6.0, 100.0]]),
        "scatter_reduce_amin": (amin, [[0.0, -100.0, 2.0, 3.0], [4.0, 5.0, 6.0, -100.0]]),
        "scatter_reduce_prod": (prod, [[0.0, 3.0, 2.0, 3.0], [4.0, 5.0, 6.0, 21.0]]),
        "index_add": (
            lambda d: flat(d).index_add(0, _ix([1, bad], d), lucid.full((2,), 10.0).to(d)),
            [0.0, 10.0, 0.0, 0.0],
        ),
        "index_copy": (
            lambda d: flat(d).index_copy(0, _ix([1, bad], d), lucid.full((2,), 10.0).to(d)),
            [0.0, 10.0, 0.0, 0.0],
        ),
        "index_fill": (
            lambda d: flat(d).index_fill(0, _ix([1, bad], d), 7.0),
            [0.0, 7.0, 0.0, 0.0],
        ),
        "index_put_": (index_put, [0.0, 10.0, 0.0, 0.0]),
    }


@pytest.mark.parametrize(
    ("name", "case"), _name_case_params(list(_scatter_cases(0)), list(_WRAPPING))
)
def test_scatter_family(device: str, name: str, case: str) -> None:
    run, answer = _scatter_cases(_WRAPPING[case])[name]
    _check(device, lambda: run(device), answer)


def _engine_full_width_cases() -> dict[str, tuple[Callable[[str], lucid.Tensor], object]]:
    """The engine ops themselves at 2**40, past every Python-side cast."""
    huge = 2**40

    def call(fn: Callable[..., object], *args: object) -> lucid.Tensor:
        return _wrap(fn(*[_unwrap(a) if isinstance(a, lucid.Tensor) else a for a in args]))

    def ones(d: str) -> lucid.Tensor:
        return lucid.full((1, 2), 10.0).to(d)

    def ix(d: str) -> lucid.Tensor:
        return _ix([[1, huge]], d)

    def row(d: str) -> lucid.Tensor:
        return lucid.arange(N, dtype=lucid.float32).reshape(1, N).to(d)

    return {
        "gather": (lambda d: call(_C_engine.gather, row(d), ix(d), 1), [[1.0, NAN]]),
        "take": (lambda d: call(_C_engine.take, row(d), _ix([huge], d)), [NAN]),
        "scatter_add": (
            lambda d: call(_C_engine.scatter_add, row(d), ix(d), ones(d), 1),
            [[0.0, 11.0, 2.0, 3.0]],
        ),
        "scatter": (
            lambda d: call(_C_engine.scatter, row(d), 1, ix(d), ones(d)),
            [[0.0, 10.0, 2.0, 3.0]],
        ),
        "scatter_amax": (
            lambda d: call(_C_engine.scatter_amax, row(d), ix(d), ones(d), 1),
            [[0.0, 10.0, 2.0, 3.0]],
        ),
    }


@pytest.mark.parametrize("name", list(_engine_full_width_cases()))
def test_engine_checks_the_full_width(device: str, name: str) -> None:
    run, answer = _engine_full_width_cases()[name]
    _check(device, lambda: run(device), answer)


@pytest.mark.parametrize("case", list(_WRAPPING))
def test_setitem(device: str, case: str) -> None:
    bad = _WRAPPING[case]
    x = lucid.zeros(N).to(device)

    def run() -> lucid.Tensor:
        x[_ix([1, bad], device)] = 5.0
        return x

    _check(device, run, [0.0, 5.0, 0.0, 0.0])


def test_scatter_leaves_its_base_alone(device_gpu_only: str) -> None:
    d = device_gpu_only
    base = _base(d)
    before = base.tolist()
    lucid.scatter(base, 1, _ix([[1, 99], [-99, 3]], d), lucid.full((2, 2), 10.0).to(d))
    lucid.scatter_add(base, 1, _ix([[1, 99], [-99, 3]], d), lucid.full((2, 2), 10.0).to(d))
    assert base.tolist() == before


# ── gradients: what an out-of-range position read, it does not send back ────


def test_gather_backward_drops_out_of_range(device: str) -> None:
    x = _base(device).requires_grad_(True)
    idx = _ix([[1, N], [-N - 1, 3]], device)
    if device == "cpu":
        with pytest.raises(IndexError):
            lucid.gather(x, 1, idx)
        return
    y = lucid.gather(x, 1, idx)
    y.backward(lucid.ones(2, 2).to(device))
    assert x.grad.tolist() == [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def test_embedding_backward_drops_out_of_range(device: str) -> None:
    w = lucid.zeros(N, 3).to(device).requires_grad_(True)
    idx = _ix([2, N, -1], device)
    if device == "cpu":
        with pytest.raises(IndexError):
            _C_engine.nn.embedding(_unwrap(w), _unwrap(idx), -1)
        return
    out = _wrap(_C_engine.nn.embedding(_unwrap(w), _unwrap(idx), -1))
    out.backward(lucid.ones(3, 3).to(device))
    assert w.grad.tolist() == [[0.0] * 3, [0.0] * 3, [1.0] * 3, [0.0] * 3]


# ── an empty axis: decided from the shapes, refused on both devices ──────────


def _empty_axis_cases() -> dict[str, Callable[[str], object]]:
    def empty(device: str) -> lucid.Tensor:
        return lucid.zeros(2, 0).to(device)

    return {
        "gather": lambda d: lucid.gather(empty(d), 1, _ix([[0], [0]], d)),
        "index_select": lambda d: lucid.index_select(empty(d), 1, _ix([0], d)),
        "take": lambda d: lucid.take(empty(d), _ix([0], d)),
        "scatter_add": lambda d: lucid.scatter_add(
            empty(d), 1, _ix([[0], [0]], d), lucid.ones(2, 1).to(d)
        ),
        "scatter": lambda d: lucid.scatter(empty(d), 1, _ix([[0], [0]], d), lucid.ones(2, 1).to(d)),
        **{
            f"scatter_reduce_{r}": (
                lambda d, r=r: lucid.scatter_reduce(
                    empty(d), 1, _ix([[0], [0]], d), lucid.zeros(2, 1).to(d), r
                )
            )
            for r in ("amax", "amin", "prod")
        },
    }


# The reference's rule: only the corner of src the index covers is
# scattered.  A src wider than the index was misplaced by the CPU reduce
# loop and a reshape error on Metal, until the four axis scatters shared one
# operand check (``axis_scatter_operands`` in Gfunc.cpp).
_WIDE_SRC_CASES = {
    "scatter_add": [[6.0, 1.0, 1.0], [1.0, -7.0, 1.0]],
    "amax": [[5.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "amin": [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0]],
    "prod": [[5.0, 1.0, 1.0], [1.0, -8.0, 1.0]],
}


@pytest.mark.parametrize("name", list(_WIDE_SRC_CASES))
def test_src_wider_than_the_index(device: str, name: str) -> None:
    base = lucid.ones(2, 3).to(device)
    idx = _ix([[0], [1]], device)
    src = lucid.tensor([[5.0, 6.0, 7.0], [-8.0, 9.0, 10.0]]).to(device)
    if name == "scatter_add":
        out = lucid.scatter_add(base, 1, idx, src)
    else:
        out = lucid.scatter_reduce(base, 1, idx, src, name)
    assert out.tolist() == _WIDE_SRC_CASES[name]


@pytest.mark.parametrize("reduce", ["amax", "amin", "prod"])
def test_scatter_reduce_index_narrower_than_base(device: str, reduce: str) -> None:
    # The CPU loop placed the index's elements by the base's strides, so an
    # index with fewer rows than the base read past the index and the src:
    # SIGBUS.
    base = lucid.full((3, N), 2.0).to(device)
    src = lucid.tensor([[7.0], [8.0]]).to(device)
    out = lucid.scatter_reduce(base, 1, _ix([[0], [1]], device), src, reduce)
    hit = {"amax": (7.0, 8.0), "amin": (2.0, 2.0), "prod": (14.0, 16.0)}[reduce]
    want = [[2.0] * N for _ in range(3)]
    want[0][0], want[1][1] = hit
    assert out.tolist() == want


def test_embedding_bag_offset_past_the_indices(device_cpu_only: str) -> None:
    # An offset past the end walked the index buffer past its end.
    d = device_cpu_only
    table = lucid.ones(N, 3).to(d)
    offsets = lucid.tensor([0, 9], dtype=lucid.int32).to(d)
    with pytest.raises(IndexError):
        _C_engine.nn.embedding_bag(
            _unwrap(table), _unwrap(_ix([1, 2], d)), _unwrap(offsets), 0, -1, False
        )


def test_class_target_is_compared_with_ignore_index_at_full_width(device: str) -> None:
    # Cut to ``int`` first, 2**32 - 100 read as the default ignore_index -100
    # and the target was silently skipped.
    logits = lucid.zeros(1, N).to(device)
    target = _ix([2**32 - 100], device)
    run = lambda: _wrap(_C_engine.nn.nll_loss(_unwrap(logits), _unwrap(target), None, 0))
    _check(device, run, [NAN])


@pytest.mark.parametrize("name", list(_empty_axis_cases()))
def test_empty_axis_is_refused_on_both_devices(device: str, name: str) -> None:
    with pytest.raises(IndexError):
        _empty_axis_cases()[name](device)


def test_empty_table(device: str) -> None:
    # A table with no rows has nothing to look up; Metal answers NaN.
    table = lucid.zeros(0, 3).to(device)
    run = lambda: _wrap(_C_engine.nn.embedding(_unwrap(table), _unwrap(_ix([0], device)), -1))
    _check(device, run, [[NAN, NAN, NAN]])


# ── Metal isolates in the graph: no host sync ────────────────────────────────


@pytest.mark.skipif(not metal_available(), reason="Metal device not available")
def test_metal_isolation_takes_no_host_sync() -> None:
    d = "metal"
    base = _base(d)
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(d)
    bad = _ix([[1, 2**40], [-N - 1, 3]], d)
    row = _ix([2, N], d)
    lucid.eval(base, table, bad, row)
    before = _C_engine.host_sync_count()
    outs = [
        lucid.gather(base, 1, bad),
        lucid.scatter_add(base, 1, bad, lucid.ones(2, 2).to(d)),
        lucid.scatter(base, 1, bad, lucid.ones(2, 2).to(d)),
        _wrap(_C_engine.nn.embedding(_unwrap(table), _unwrap(row), -1)),
        _wrap(_C_engine.nn.one_hot(_unwrap(row), N)),
    ]
    lucid.eval(*outs)
    assert _C_engine.host_sync_count() == before


# ── strict mode (LCD-892): Metal refuses, in the CPU's words ─────────────────
#
# Policy B is the default and the tests above pin it.  ``LUCID_STRICT_INDEX``
# is the opt-in way back to the CPU's IndexError: the gates read the check
# back to the host, so a bad index raises where it was made instead of
# becoming a NaN far away.  The message is the CPU's, word for word.


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_STRICT_INDEX", "1")


def _refusal(run: Callable[[str], object], device: str) -> str:
    with pytest.raises(IndexError) as info:
        run(device)
    return str(info.value)


@pytest.mark.parametrize(
    ("name", "case"), _name_case_params(list(_gather_cases(0)), list(_WRAPPING))
)
def test_strict_gather_family(device_gpu_only: str, strict: None, name: str, case: str) -> None:
    run, _ = _gather_cases(_WRAPPING[case])[name]
    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


# The class losses name the op by what called the engine, so for them only
# the refusal itself is pinned here; ``test_strict_class_losses_*`` pins
# the words through the public functions.
_ROWS_WORDED_LIKE_THE_CPU = {"embedding", "embedding_bag", "one_hot"}


@pytest.mark.parametrize("case", list(_NON_WRAPPING))
@pytest.mark.parametrize("name", list(_row_cases(0)))
def test_strict_row_and_class_family(
    device_gpu_only: str, strict: None, name: str, case: str
) -> None:
    run, _ = _row_cases(_NON_WRAPPING[case])[name]
    metal = _refusal(run, device_gpu_only)
    if name in _ROWS_WORDED_LIKE_THE_CPU:
        assert metal == _refusal(run, "cpu")


@pytest.mark.parametrize(
    ("name", "case"), _name_case_params(list(_scatter_cases(0)), list(_WRAPPING))
)
def test_strict_scatter_family(device_gpu_only: str, strict: None, name: str, case: str) -> None:
    run, _ = _scatter_cases(_WRAPPING[case])[name]
    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


@pytest.mark.parametrize("name", list(_engine_full_width_cases()))
def test_strict_engine_ops_at_full_width(device_gpu_only: str, strict: None, name: str) -> None:
    run, _ = _engine_full_width_cases()[name]
    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


@pytest.mark.parametrize("case", list(_WRAPPING))
def test_strict_setitem(device_gpu_only: str, strict: None, case: str) -> None:
    bad = _WRAPPING[case]

    def run(device: str) -> lucid.Tensor:
        x = lucid.zeros(N).to(device)
        x[_ix([1, bad], device)] = 5.0
        return x

    _refusal(run, device_gpu_only)


def test_strict_embedding_module_says_what_the_cpu_says(device_gpu_only: str, strict: None) -> None:
    import lucid.nn as nn

    def run(device: str) -> lucid.Tensor:
        return nn.Embedding(N, 3).to(device)(_ix([9], device))

    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


@pytest.mark.parametrize(
    "name", ["cross_entropy", "nll_loss", "multi_margin_loss", "multilabel_margin_loss"]
)
def test_strict_class_losses_say_what_the_cpu_says(
    device_gpu_only: str, strict: None, name: str
) -> None:
    import lucid.nn.functional as F

    def run(device: str) -> lucid.Tensor:
        logits = lucid.tensor([[0.0, 1.0, 2.0, 3.0]]).to(device)
        target = _ix([[9, -1, 0, 0]] if name == "multilabel_margin_loss" else [9], device)
        return getattr(F, name)(logits, target)

    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


def test_strict_class_loss_ignores_the_ignore_index(device_gpu_only: str, strict: None) -> None:
    import lucid.nn.functional as F

    metal = F.cross_entropy(lucid.zeros(2, N).to(device_gpu_only), _ix([1, -100], device_gpu_only))
    cpu = F.cross_entropy(lucid.zeros(2, N), _ix([1, -100], "cpu"))
    assert metal.item() == pytest.approx(cpu.item())


def test_strict_one_hot_functional(device_gpu_only: str, strict: None) -> None:
    import lucid.nn.functional as F

    def run(device: str) -> lucid.Tensor:
        return F.one_hot(_ix([1, 9], device), N)

    assert _refusal(run, device_gpu_only) == _refusal(run, "cpu")


def test_strict_leaves_valid_calls_alone(device: str, strict: None) -> None:
    """In range, strict mode answers what policy B answers: the same values,
    negative wrap-around included, and no refusal."""
    import lucid.nn.functional as F

    grid = lucid.arange(2 * N, dtype=lucid.float32).reshape(2, N).to(device)
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)
    row = _ix([3, 0, -1, -N], device)  # negatives wrap for a gather
    assert lucid.index_select(grid, 1, row).tolist() == [[3.0, 0.0, 3.0, 0.0], [7.0, 4.0, 7.0, 4.0]]
    assert lucid.gather(grid, 1, _ix([[3, -1], [0, -N]], device)).tolist() == [
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    assert lucid.take(grid, _ix([7, -1], device)).tolist() == [7.0, 7.0]
    assert grid[_ix([1, -1], device)].tolist() == grid[_ix([1, 1], device)].tolist()
    assert F.embedding(_ix([[3, 0], [1, 1]], device), table).shape == (2, 2, 3)
    assert F.one_hot(_ix([0, 3], device), N).tolist() == [[1, 0, 0, 0], [0, 0, 0, 1]]
    summed = lucid.scatter_add(grid, 1, _ix([[1], [-1]], device), lucid.ones(2, 1).to(device))
    assert summed.tolist() == [[0.0, 2.0, 2.0, 3.0], [4.0, 5.0, 6.0, 8.0]]


def test_strict_rows_never_wrap(device: str, strict: None) -> None:
    # -1 is the last element for a gather and no row at all for a table.
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)
    with pytest.raises(IndexError):
        _C_engine.nn.embedding(_unwrap(table), _unwrap(_ix([-1], device)), -1)


def test_strict_padding_idx_keeps_its_behaviour(device: str, strict: None) -> None:
    import lucid.nn.functional as F

    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)
    padded = F.embedding(_ix([1, 2, 1], device), table, padding_idx=1)
    plain = F.embedding(_ix([1, 2, 1], device), table)
    assert padded.tolist() == plain.tolist()
    with pytest.raises(IndexError):
        F.embedding(_ix([1, N], device), table, padding_idx=1)


def test_strict_embedding_backward_with_padding_idx(device: str, strict: None) -> None:
    import lucid.nn.functional as F

    w = lucid.zeros(N, 3).to(device).requires_grad_(True)
    F.embedding(_ix([1, 2, 1], device), w, padding_idx=1).sum().backward()
    assert w.grad.tolist() == [[0.0] * 3, [0.0] * 3, [1.0] * 3, [0.0] * 3]


@pytest.mark.parametrize(
    ("value", "raises"),
    [("1", True), ("true", True), ("yes", True), ("0", False), ("", False), ("0abc", False)],
)
def test_strict_switch_reads_the_same_in_both_layers(
    device_gpu_only: str, monkeypatch: pytest.MonkeyPatch, value: str, raises: bool
) -> None:
    # The Python guards and the engine each read LUCID_STRICT_INDEX; they
    # must agree, for an engine op the guards never see and for one they do.
    import lucid.nn.functional as F

    monkeypatch.setenv("LUCID_STRICT_INDEX", value)
    d = device_gpu_only
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(d)
    engine_op = lambda: _wrap(_C_engine.nn.embedding(_unwrap(table), _unwrap(_ix([N], d)), -1))
    guarded_op = lambda: F.embedding(_ix([N], d), table)
    for call in (engine_op, guarded_op):
        if raises:
            with pytest.raises(IndexError):
                call()
        else:
            assert all(math.isnan(v) for v in call().tolist()[0])


# Compiled programs.  The call that traces a program runs eagerly, so strict
# mode refuses on it; a replay runs the graph, where an index is isolated
# in-graph (``compile/OpEmitters/_IndexBounds.h``) and nothing reads it back.
# A call that falls back to eager (it needs gradients) is checked every time.


def _compiled_embedding(device: str) -> Callable[[lucid.Tensor], lucid.Tensor]:
    import lucid.nn.functional as F

    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(device)
    return lucid.compile(lambda x: F.embedding(x, table))


def test_strict_compiled_trace_call_refuses(device_gpu_only: str, strict: None) -> None:
    run = _compiled_embedding(device_gpu_only)
    with lucid.no_grad(), pytest.raises(IndexError, match="index 9 is out of range"):
        run(_ix([0, 9], device_gpu_only))


@pytest.mark.xfail(
    strict=True,
    reason="LCD-892: a compiled replay on Metal does not read its indices back; "
    "the check has to ride the graph (export role)",
)
def test_strict_compiled_replay_refuses(device_gpu_only: str, strict: None) -> None:
    run = _compiled_embedding(device_gpu_only)
    with lucid.no_grad():
        run(_ix([0, 1], device_gpu_only))
        with pytest.raises(IndexError):
            run(_ix([0, 9], device_gpu_only))


@pytest.mark.filterwarnings("ignore:lucid.compile:UserWarning")
def test_strict_compiled_call_that_needs_gradients_refuses(
    device_gpu_only: str, strict: None
) -> None:
    import lucid.nn as nn

    run = lucid.compile(nn.Embedding(N, 3).to(device_gpu_only))
    run(_ix([0, 1], device_gpu_only))
    with pytest.raises(IndexError, match="index 9 is out of range"):
        run(_ix([0, 9], device_gpu_only))


def test_strict_is_off_unless_asked(device_gpu_only: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUCID_STRICT_INDEX", raising=False)
    d = device_gpu_only
    table = lucid.arange(N * 3, dtype=lucid.float32).reshape(N, 3).to(d)
    out = lucid.index_select(table, 0, _ix([2, N], d))
    assert _same(out.tolist(), [[6.0, 7.0, 8.0], [NAN, NAN, NAN]])
