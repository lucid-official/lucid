"""Every op-matrix case on an empty input and on a strided view, CPU and Metal.

The matrices run each case at one dense, non-empty shape.  This runs the
same recipes on what a real batch can also be:

* **strided** — the same values in transposed storage, every other element
  of a wider buffer, and at an offset into a larger one.  Must answer as
  the dense input does.  (1,234 runs; none differed when this was added.)
* **empty** — a leading axis of length 0, as an empty batch is.  Must
  answer with numpy's shape, or raise where numpy raises.

The empty sweep's first run found ``cumsum`` and ``cummax`` dying of
SIGSEGV on the CPU (they read element 0 of an empty axis); ``max``, ``min``,
``argmax`` and ``argmin`` answering ±inf or an index into nothing on the
CPU where the reference refuses; Metal raising MLX's "Cannot max reduce
zero size array" for a max-pool, ``logsumexp`` or ``argmax`` over an empty
batch, whose answer is just empty; Metal gathering "from" an empty axis;
and ``scatter_add`` taking an index larger than its operands — a crash on
Metal, an out-of-bounds read and write on the CPU.
"""

import math

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test.unit.compile import _op_matrix as M
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.ops._numpy_oracle import REFS
from lucid.test.unit.ops.test_numpy_oracle import _Consts


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


DEVICES = ["cpu"] + (["metal"] if _metal_ok() else [])

# Where numpy's answer for an empty input is not the reference framework's.
EMPTY_DIVERGES = {
    # np.split(x, 2) makes two pieces of any length; the reference's
    # split(x, 2) makes pieces of length 2 — one, here, of length 0.
    "split": "numpy splits into a count, the reference by a size",
    # The oracle's own scan indexes element 0 of the empty axis.
    "cummax": "the numpy oracle cannot scan an empty axis",
    # The oracle reshapes (0, 4, 5) into groups with -1, which numpy refuses.
    "group_norm": "the numpy oracle cannot reshape an empty batch",
}


#: Cases that cannot copy, and the layouts whose strides cannot step through
#: the case's shape: ``view`` refuses those, as the reference does, and reads
#: the rest in place.  ``view(2, 12)`` regroups a transpose's rows; every
#: other element of a wider buffer is one chunk of strides (LCD-390).
STRICT_VIEWS = {"view": frozenset({"transposed"})}


def _cases() -> list[str]:
    return [c.name for c in M.CASES if not c.random]


def _layouts(x: lucid.Tensor) -> list[tuple[str, lucid.Tensor]]:
    """The same values as ``x`` in three non-contiguous layouts."""
    out = []
    if x.ndim >= 2:
        out.append(("transposed", x.mT.contiguous().mT))
    out.append(("stepped", lucid.stack([x, lucid.zeros_like(x)], dim=-1)[..., 0]))
    if x.ndim and x.shape[0]:
        out.append(("offset", lucid.cat([x[:1], x], dim=0)[1:]))
    return [(tag, v) for tag, v in out if not v.is_contiguous()]


def _differs(got: object, want: object) -> str:
    g, w = M._leaves(got), M._leaves(want)
    if len(g) != len(w):
        return f"{len(g)} outputs, want {len(w)}"
    for i, (a, b) in enumerate(zip(g, w)):
        an, bn = a.numpy(), b.numpy()
        if an.shape != bn.shape or an.dtype != bn.dtype:
            return f"out[{i}] {an.dtype}{an.shape}, want {bn.dtype}{bn.shape}"
        if np.issubdtype(bn.dtype, np.floating):
            if not np.allclose(an, bn, rtol=1e-5, atol=1e-6, equal_nan=True):
                return f"out[{i}] values differ"
        elif not np.array_equal(an, bn):
            return f"out[{i}] values differ"
    return ""


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", _cases())
def test_a_strided_input_answers_as_a_dense_one(name: str, device: str) -> None:
    case = M.CASE_BY_NAME[name]
    with M.on_device(device):
        for dtype in case.dtypes:
            if M.refusal(name, dtype, device) is not None:
                continue
            x = M.make_input(case.kind, dtype, case.shape, 1).to(device)
            want = case.fn(x)
            for tag, view in _layouts(x):
                if tag in STRICT_VIEWS.get(name, ()):
                    # ``view`` reads its input in place or refuses, as the
                    # reference does, on Metal too (LCD-513); ``reshape`` is
                    # the copying spelling.
                    with pytest.raises(RuntimeError, match="use .reshape"):
                        case.fn(view)
                    continue
                why = _differs(case.fn(view), want)
                assert not why, f"{dtype} {tag}: {why}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", [n for n in _cases() if M.CASE_BY_NAME[n].shape])
def test_an_empty_batch_answers_as_numpy_does(name: str, device: str) -> None:
    case = M.CASE_BY_NAME[name]
    ref = REFS.get(name)
    shape = (0, *case.shape[1:])
    with M.on_device(device):
        for dtype in case.dtypes:
            if M.refusal(name, dtype, device) is not None:
                continue
            x = M.make_input(case.kind, dtype, case.shape, 1).to(device)
            case.fn(x)  # the recipe's constants, and proof the case runs
            empty = lucid.zeros(*shape, dtype=x.dtype).to(device)
            try:
                got: list[tuple[int, ...]] | None = [
                    tuple(t.shape) for t in M._leaves(case.fn(empty))
                ]
            except Exception:  # noqa: BLE001 — a refusal is an answer here
                got = None
            if ref is None or name in EMPTY_DIVERGES:
                continue  # it ran without crashing, which is the point
            try:
                with np.errstate(all="ignore"):
                    r = ref(empty.to("cpu").numpy(), _Consts())
                want: list[tuple[int, ...]] | None = [
                    tuple(np.shape(a)) for a in (r if isinstance(r, tuple) else (r,))
                ]
            except Exception:  # noqa: BLE001 — numpy refused
                want = None
            if want is None:
                assert got is None, f"{dtype}: answered {got} where numpy refuses"
            else:
                assert got == want, f"{dtype}: {got}, numpy {want}"


def test_the_divergence_table_names_real_cases() -> None:
    for name in EMPTY_DIVERGES:
        assert name in M.CASE_BY_NAME, name


# ── the fixes, one by one ────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax", "cummin"])
def test_a_scan_over_an_empty_axis_is_empty(op: str, device: str) -> None:
    for shape, dim in (((0, 6), 0), ((4, 0), 1), ((0,), 0), ((3, 0, 2), 1)):
        x = lucid.zeros(*shape).to(device)
        out = getattr(lucid, op)(x, dim)
        values = out[0] if isinstance(out, tuple) else out
        assert tuple(values.shape) == shape


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["max", "min", "argmax", "argmin"])
def test_an_extreme_over_an_empty_axis_is_refused(op: str, device: str) -> None:
    x = lucid.zeros(0, 6).to(device)
    fn = getattr(lucid, op)
    with pytest.raises(IndexError, match="reduction dim 0 to have non-zero size"):
        fn(x, dim=0)
    # The full reduction has no dimension to blame.
    with pytest.raises((RuntimeError, IndexError), match="numel\\(\\) == 0"):
        fn(x)
    # A non-empty reduced axis of an empty tensor has an empty answer.
    out = fn(x, dim=1)
    values = out[0] if isinstance(out, tuple) else out
    assert tuple(values.shape) == (0,)


@pytest.mark.parametrize("device", DEVICES)
def test_an_empty_batch_pools_and_reduces_on_every_device(device: str) -> None:
    import lucid.nn.functional as F

    x = lucid.zeros(0, 3, 8, 8).to(device)
    assert tuple(F.max_pool2d(x, 2).shape) == (0, 3, 4, 4)
    values, indices = F.max_pool2d(x, 2, return_indices=True)
    assert tuple(values.shape) == tuple(indices.shape) == (0, 3, 4, 4)
    assert tuple(lucid.logsumexp(lucid.zeros(0, 6).to(device), dim=1).shape) == (0,)
    assert lucid.linalg.norm(lucid.zeros(0, 6).to(device)).item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_argmax_without_a_dim_indexes_the_flattened_tensor(device: str) -> None:
    x = lucid.tensor([[1.0, 9.0, 3.0], [4.0, 5.0, 8.0]]).to(device)
    assert x.argmax().item() == 1 and x.argmin().item() == 0
    assert lucid.argmax(x).shape == () and lucid.argmax(x).item() == 1
    assert tuple(x.argmax(keepdim=True).shape) == (1, 1)
    assert x.argmax(dim=1).tolist() == [1, 2]
    assert lucid.tensor(7.0).to(device).argmax().item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_a_gather_from_an_empty_axis_is_out_of_range(device: str) -> None:
    empty = lucid.zeros(0, 6).to(device)
    index = lucid.tensor([0]).to(device)
    for read in (
        lambda: empty[index],
        lambda: lucid.index_select(empty, 0, index),
        lambda: lucid.take(empty, index),
    ):
        with pytest.raises(IndexError):
            read()
    # Gathering nothing from nothing is fine.
    none = lucid.tensor([], dtype=lucid.int64).to(device)
    assert tuple(lucid.index_select(empty, 0, none).shape) == (0, 6)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_add_refuses_an_index_larger_than_its_operands(device: str) -> None:
    index = lucid.zeros(4, 6, dtype=lucid.int64).to(device)
    with pytest.raises(RuntimeError, match="no larger than self"):
        lucid.scatter_add(
            lucid.zeros(2, 6).to(device), 1, index, lucid.ones(4, 6).to(device)
        )
    with pytest.raises(RuntimeError, match="no larger than src"):
        lucid.scatter_add(
            lucid.zeros(4, 6).to(device), 1, index, lucid.ones(2, 6).to(device)
        )
    with pytest.raises(RuntimeError, match="no larger than self"):
        lucid.scatter_reduce(
            lucid.zeros(2, 6).to(device), 1, index, lucid.ones(4, 6).to(device), "sum"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_an_index_smaller_than_its_operands_scatters_and_gathers(device: str) -> None:
    # The reference's rule: the index may be shorter than self on every axis
    # but ``dim`` and shorter than src on every axis.  The CPU walked base's
    # extents and read past a shorter index and src — stale memory in the
    # answer — and Metal refused the call with a broadcast error.
    idx = [[0, 2, 5], [1, 4, 3]]
    src = np.arange(15.0, dtype=np.float32).reshape(3, 5)
    want = np.zeros((4, 6), np.float32)
    for r, row in enumerate(idx):
        for c, t in enumerate(row):
            want[r, t] += src[r, c]
    index = lucid.tensor(idx).to(device)
    added = lucid.scatter_add(
        lucid.zeros(4, 6).to(device), 1, index, lucid.tensor(src).to(device)
    )
    np.testing.assert_array_equal(added.numpy(), want)
    written = lucid.scatter(
        lucid.zeros(4, 6).to(device), 1, index, lucid.tensor(src).to(device)
    )
    np.testing.assert_array_equal(written.numpy(), want)  # no duplicates: same answer

    table = np.arange(24.0, dtype=np.float32).reshape(4, 6)
    x = lucid.tensor(table).to(device).requires_grad_()
    picked = lucid.gather(x, 1, index)
    np.testing.assert_array_equal(
        picked.numpy(), np.take_along_axis(table[:2], np.array(idx), 1)
    )
    (picked * 3.0).sum().backward()
    grad = np.zeros((4, 6), np.float32)
    for r, row in enumerate(idx):
        grad[r, row] += 3.0
    np.testing.assert_array_equal(x.grad.numpy(), grad)

    # A src larger than the index sends its gradient to the corner it gave.
    s = lucid.tensor(src).to(device).requires_grad_()
    (
        lucid.scatter_add(lucid.zeros(4, 6).to(device), 1, index, s) * 2.0
    ).sum().backward()
    corner = np.zeros((3, 5), np.float32)
    corner[:2, :3] = 2.0
    np.testing.assert_array_equal(s.grad.numpy(), corner)


@pytest.mark.parametrize("device", DEVICES)
def test_index_copy_refuses_a_source_of_the_wrong_width(device: str) -> None:
    base = lucid.zeros(4, 6).to(device)
    rows = lucid.tensor([3, 1]).to(device)
    with pytest.raises(_C_engine.ShapeMismatch):
        lucid.index_copy(base, 0, rows, lucid.ones(2, 3).to(device))
    copied = lucid.index_copy(base, 0, rows, lucid.ones(2, 6).to(device))
    assert copied.sum(dim=1).tolist() == [0.0, 6.0, 0.0, 6.0]


@pytest.mark.parametrize("device", DEVICES)
def test_ops_built_on_max_still_answer_an_empty_input(device: str) -> None:
    # Refusing ``max`` of nothing reached every composite that took one on
    # the way: each has an answer for an empty input that needs no maximum.
    from lucid.nn.utils import clip_grad_norm_

    empty_rows = lucid.zeros(0, 6).to(device)
    assert lucid.logsumexp(empty_rows, dim=0).tolist() == [-math.inf] * 6
    assert lucid.logsumexp(lucid.zeros(0).to(device), dim=0).item() == -math.inf
    assert lucid.histc(lucid.tensor([]).to(device), bins=4).tolist() == [0.0] * 4
    assert lucid.linalg.matrix_rank(lucid.zeros(0, 3).to(device)).item() == 0
    assert tuple(lucid.linalg.matrix_exp(lucid.zeros(0, 0).to(device)).shape) == (0, 0)

    empty = lucid.nn.Parameter(lucid.zeros(0).to(device))
    full = lucid.nn.Parameter(lucid.ones(3).to(device))
    (full.sum() * 2.0 + empty.sum()).backward()
    assert float(clip_grad_norm_([empty, full], 10.0, norm_type=math.inf)) == 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_eye_takes_an_empty_side(device: str) -> None:
    # 0 was read as "no width given" (3 x 0 came back 3 x 3), and MLX
    # refuses an identity with an empty side outright.
    for args, shape in (
        ((0,), (0, 0)),
        ((0, 3), (0, 3)),
        ((3, 0), (3, 0)),
        ((2, 4), (2, 4)),
    ):
        assert tuple(lucid.eye(*args, device=device).shape) == shape
