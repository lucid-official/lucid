"""Contract guards for view storage geometry across CPU and Metal (LCD-517 Q1).

Every view operation — indexing, slicing, narrowing, splitting, transposing,
diagonal/unfold/expand, reshaping, detaching, ``.data``, ``as_strided``, and
real ``conj`` — across ``first``, ``middle``, ``last``, and ``empty`` positions,
as well as seeded random view chains of depth 1–4 from dense ``(4, 6)`` and
``(2, 3, 4)`` roots, is checked against four storage-frame invariants:

1. ``test_offset_counts_from_the_root``: a view's ``storage_offset()`` counts
   from the root allocation's first element (`v.data_ptr() - root.data_ptr()
   == v.storage_offset() * itemsize` on CPU, and Metal's offset/stride match
   the CPU chain's true root-relative geometry).
2. ``test_as_strided_reads_the_root``: ``v.as_strided((n,), (1,), 0)`` reads the
   root's first ``n`` elements, ``v.as_strided((root.numel(),), (1,))`` spans
   the whole root from an offset-0 view, and ``v.as_strided(v.shape,
   v.stride())`` reproduces ``v``.
3. ``test_untyped_storage_is_the_roots``: ``v.untyped_storage()`` has the same
   ``data_ptr()`` and ``size() == root.nbytes`` as the root, with
   ``v.data_ptr() == v.untyped_storage().data_ptr() + v.storage_offset() *
   itemsize`` on CPU.
4. ``test_save_load_keeps_geometry``: saving and loading ``{"root": root,
   "v": v}`` preserves ``v``'s values, ``storage_offset()``, moving strides,
   and shared-storage aliasing with ``root``.
"""

from collections.abc import Callable
import io
import math
import random
from typing import NamedTuple

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

type _View = Callable[[lucid.Tensor], lucid.Tensor]

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)


def _moving(shape: tuple[int, ...], stride: tuple[int, ...]) -> tuple[int, ...]:
    if 0 in shape:
        return ()
    return tuple(st for n, st in zip(shape, stride, strict=True) if n != 1)


def _cpu_offset(view: lucid.Tensor, root: lucid.Tensor) -> int:
    return (view.data_ptr() - root.data_ptr()) // view.element_size()


def _root_2d(device: str) -> lucid.Tensor:
    return lucid.arange(24.0, device=device).reshape(4, 6)


class _Cell(NamedTuple):
    make: _View
    expected_offset: int
    #: CPU ``make_block_view`` narrows ``CpuStorage`` to the slice's span.
    cpu_narrowed: bool


#: Grid of (view op x position) on a dense (4, 6) root (24 float32 elements).
#: Positions cover first, middle, last, and empty pieces across the view ops.
GRID: dict[str, _Cell] = {
    # int index
    "int-first": _Cell(lambda t: t[0], 0, True),
    "int-middle": _Cell(lambda t: t[1], 6, True),
    "int-last": _Cell(lambda t: t[-1], 18, True),
    # slice (dim 0 and dim 1)
    "slice-first": _Cell(lambda t: t[:1], 0, True),
    "slice-middle": _Cell(lambda t: t[1:3], 6, True),
    "slice-last": _Cell(lambda t: t[-1:], 18, True),
    "slice-empty": _Cell(lambda t: t[3:1], 18, True),
    "slice-col-first": _Cell(lambda t: t[:, :2], 0, False),
    "slice-col-middle": _Cell(lambda t: t[:, 2:4], 2, False),
    "slice-col-last": _Cell(lambda t: t[:, -2:], 4, False),
    "slice-col-empty": _Cell(lambda t: t[:, 3:1], 3, False),
    # stepped slice
    "step-first": _Cell(lambda t: t[:3:2], 0, True),
    "step-middle": _Cell(lambda t: t[1::2], 6, True),
    "step-last": _Cell(lambda t: t[2::2], 12, True),
    "step-col-middle": _Cell(lambda t: t[:, 1::2], 1, False),
    # narrow
    "narrow-first": _Cell(lambda t: t.narrow(0, 0, 1), 0, True),
    "narrow-middle": _Cell(lambda t: t.narrow(0, 1, 2), 6, True),
    "narrow-last": _Cell(lambda t: t.narrow(0, 3, 1), 18, True),
    "narrow-empty": _Cell(lambda t: t.narrow(0, 2, 0), 12, True),
    "narrow-col-middle": _Cell(lambda t: t.narrow(1, 2, 3), 2, False),
    # unbind[k]
    "unbind-first": _Cell(lambda t: t.unbind(0)[0], 0, True),
    "unbind-middle": _Cell(lambda t: t.unbind(0)[1], 6, True),
    "unbind-last": _Cell(lambda t: t.unbind(0)[-1], 18, True),
    "unbind-col-middle": _Cell(lambda t: t.unbind(1)[2], 2, False),
    # split(n)[k]
    "split-n-first": _Cell(lambda t: t.split(1, 0)[0], 0, True),
    "split-n-middle": _Cell(lambda t: t.split(1, 0)[1], 6, True),
    "split-n-last": _Cell(lambda t: t.split(1, 0)[-1], 18, True),
    "split-n-col-middle": _Cell(lambda t: t.split(2, 1)[1], 2, False),
    # split(sizes)[k]
    "split-sizes-first": _Cell(lambda t: t.split([1, 2, 1, 0], 0)[0], 0, True),
    "split-sizes-middle": _Cell(lambda t: t.split([1, 2, 1, 0], 0)[1], 6, True),
    "split-sizes-last": _Cell(lambda t: t.split([1, 2, 1, 0], 0)[2], 18, True),
    "split-sizes-empty": _Cell(lambda t: t.split([1, 2, 1, 0], 0)[3], 24, True),
    # chunk[k]
    "chunk-first": _Cell(lambda t: t.chunk(4, 0)[0], 0, True),
    "chunk-middle": _Cell(lambda t: t.chunk(4, 0)[1], 6, True),
    "chunk-last": _Cell(lambda t: t.chunk(4, 0)[-1], 18, True),
    "chunk-col-middle": _Cell(lambda t: t.chunk(3, 1)[1], 2, False),
    # tensor_split / hsplit / vsplit
    "tensor_split-first": _Cell(lambda t: t.tensor_split([1, 3, 3], 0)[0], 0, True),
    "tensor_split-middle": _Cell(lambda t: t.tensor_split([1, 3, 3], 0)[1], 6, True),
    "tensor_split-last": _Cell(lambda t: t.tensor_split([1, 3, 3], 0)[3], 18, True),
    "tensor_split-empty": _Cell(lambda t: t.tensor_split([1, 3, 3], 0)[2], 18, True),
    "hsplit-first": _Cell(lambda t: t.hsplit([2, 4, 4])[0], 0, False),
    "hsplit-middle": _Cell(lambda t: t.hsplit([2, 4, 4])[1], 2, False),
    "hsplit-last": _Cell(lambda t: t.hsplit([2, 4, 4])[3], 4, False),
    "hsplit-empty": _Cell(lambda t: t.hsplit([2, 4, 4])[2], 4, False),
    "vsplit-first": _Cell(lambda t: t.vsplit([1, 3, 3])[0], 0, True),
    "vsplit-middle": _Cell(lambda t: t.vsplit([1, 3, 3])[1], 6, True),
    "vsplit-last": _Cell(lambda t: t.vsplit([1, 3, 3])[3], 18, True),
    "vsplit-empty": _Cell(lambda t: t.vsplit([1, 3, 3])[2], 18, True),
    # list(t)[k]
    "list-first": _Cell(lambda t: list(t)[0], 0, True),
    "list-middle": _Cell(lambda t: list(t)[1], 6, True),
    "list-last": _Cell(lambda t: list(t)[-1], 18, True),
    # t / mT / permute / transpose / movedim
    "t-whole": _Cell(lambda t: t.t(), 0, False),
    "t-middle": _Cell(lambda t: t.t()[2], 2, False),
    "t-last": _Cell(lambda t: t.t()[-1], 5, False),
    "mT-middle": _Cell(lambda t: t.mT[1:3], 1, False),
    "permute-middle": _Cell(lambda t: t.permute(1, 0)[2:4], 2, False),
    "transpose-middle": _Cell(lambda t: t.transpose(0, 1)[1:4], 1, False),
    "movedim-middle": _Cell(lambda t: t.movedim(0, -1)[3], 3, False),
    # diagonal
    "diagonal-first": _Cell(lambda t: t.diagonal(0), 0, False),
    "diagonal-middle": _Cell(lambda t: t.diagonal(2), 2, False),
    "diagonal-last": _Cell(lambda t: t.diagonal(-2), 12, False),
    "diagonal-empty": _Cell(lambda t: t.diagonal(10), 0, False),
    # unfold
    "unfold-first": _Cell(lambda t: t.unfold(0, 2, 1)[0], 0, False),
    "unfold-middle": _Cell(lambda t: t.unfold(0, 2, 1)[1], 6, False),
    "unfold-last": _Cell(lambda t: t.unfold(0, 2, 1)[-1], 12, False),
    "unfold-col": _Cell(lambda t: t.unfold(1, 3, 2), 0, False),
    # expand
    "expand-first": _Cell(lambda t: t[:1].expand(3, 6), 0, True),
    "expand-middle": _Cell(lambda t: t[1:2].expand(3, 6), 6, True),
    "expand-last": _Cell(lambda t: t[-1:].expand(3, 6), 18, True),
    "expand-empty": _Cell(lambda t: t[2:2].expand(0, 6), 12, True),
    "expand-col": _Cell(lambda t: t[:, :1].expand(4, 3), 0, False),
    # view / reshape / flatten / unflatten
    "view-first": _Cell(lambda t: t[:2].view(-1), 0, True),
    "view-middle": _Cell(lambda t: t[1:3].view(-1), 6, True),
    "view-last": _Cell(lambda t: t[-1:].view(-1), 18, True),
    "reshape-first": _Cell(lambda t: t[:2].reshape(3, 4), 0, True),
    "reshape-middle": _Cell(lambda t: t[1:3].reshape(3, 4), 6, True),
    "reshape-last": _Cell(lambda t: t[-1:].reshape(2, 3), 18, True),
    "flatten-middle": _Cell(lambda t: t[1:3].flatten(), 6, True),
    "unflatten-middle": _Cell(lambda t: t.unflatten(0, (2, 2))[1], 12, True),
    # squeeze / unsqueeze
    "squeeze-first": _Cell(lambda t: t[:1].squeeze(0), 0, True),
    "squeeze-middle": _Cell(lambda t: t[1:2].squeeze(0), 6, True),
    "squeeze-last": _Cell(lambda t: t[-1:].squeeze(0), 18, True),
    "unsqueeze-middle": _Cell(lambda t: t[1:3].unsqueeze(1), 6, True),
    # detach / .data
    "detach-first": _Cell(lambda t: t[0].detach(), 0, True),
    "detach-middle": _Cell(lambda t: t[1].detach(), 6, True),
    "detach-last": _Cell(lambda t: t[-1].detach(), 18, True),
    "data-first": _Cell(lambda t: t[0].data, 0, True),
    "data-middle": _Cell(lambda t: t[1].data, 6, True),
    "data-last": _Cell(lambda t: t[-1].data, 18, True),
    # as_strided(own)
    "as_strided-first": _Cell(lambda t: t.as_strided((2, 3), (6, 1), 0), 0, False),
    "as_strided-middle": _Cell(lambda t: t.as_strided((2, 3), (6, 1), 6), 6, False),
    "as_strided-last": _Cell(lambda t: t.as_strided((1, 3), (6, 1), 21), 21, False),
    "as_strided-empty": _Cell(lambda t: t.as_strided((0, 3), (6, 1), 6), 6, False),
    "as_strided-on-row": _Cell(lambda t: t[1:].as_strided((2, 3), (6, 1)), 6, True),
    # conj(real)
    "conj-first": _Cell(lambda t: lucid.conj(t[0]), 0, True),
    "conj-middle": _Cell(lambda t: lucid.conj(t[1]), 6, True),
    "conj-last": _Cell(lambda t: lucid.conj(t[-1]), 18, True),
    # chained slice of slice
    "row-of-rows": _Cell(lambda t: t[1:][1], 12, True),
}


def _grid_params() -> list[object]:
    params: list[object] = []
    for name in GRID:
        for device in _DEVICES:
            params.append(pytest.param(name, device, id=f"{name}-{device}"))
    return params


@pytest.mark.parametrize(("name", "device"), _grid_params())
def test_offset_counts_from_the_root(name: str, device: str) -> None:
    cell = GRID[name]
    root = _root_2d(device)
    v = cell.make(root)
    cpu_root = _root_2d("cpu")
    cpu_v = cell.make(cpu_root)
    assert v.shape == cpu_v.shape
    assert _moving(v.shape, v.stride()) == _moving(cpu_v.shape, cpu_v.stride())
    if device == "cpu":
        if v.numel() > 0:
            assert (
                v.data_ptr() - root.data_ptr() == v.storage_offset() * v.element_size()
            )
        assert v.storage_offset() == cell.expected_offset
    else:
        if cpu_v.numel() > 0:
            assert v.storage_offset() == _cpu_offset(cpu_v, cpu_root)
        assert v.storage_offset() == cell.expected_offset


@pytest.mark.parametrize(("name", "device"), _grid_params())
def test_as_strided_reads_the_root(name: str, device: str) -> None:
    cell = GRID[name]
    root = _root_2d(device)
    v = cell.make(root)
    assert v.as_strided(v.shape, v.stride()).tolist() == v.tolist()
    assert v.as_strided((3,), (1,), 0).tolist() == root.reshape(-1)[:3].tolist()
    if cell.expected_offset == 0 and v.numel() > 0:
        assert v.as_strided((root.numel(),), (1,)).tolist() == root.reshape(-1).tolist()


@pytest.mark.parametrize(("name", "device"), _grid_params())
def test_untyped_storage_is_the_roots(name: str, device: str) -> None:
    cell = GRID[name]
    root = _root_2d(device)
    v = cell.make(root)
    assert v.untyped_storage().data_ptr() == root.untyped_storage().data_ptr()
    assert v.untyped_storage().size() == root.nbytes
    if device == "cpu" and v.numel() > 0:
        assert (
            v.data_ptr()
            == v.untyped_storage().data_ptr() + v.storage_offset() * v.element_size()
        )


@pytest.mark.parametrize(("name", "device"), _grid_params())
def test_save_load_keeps_geometry(name: str, device: str) -> None:
    cell = GRID[name]
    root = _root_2d(device)
    v = cell.make(root)
    buf = io.BytesIO()
    lucid.save({"root": root, "v": v}, buf)
    buf.seek(0)
    loaded = lucid.load(buf)
    assert isinstance(loaded, dict)
    lroot, lv = loaded["root"], loaded["v"]
    assert lroot.tolist() == root.tolist()
    assert lv.shape == v.shape
    assert lv.tolist() == v.tolist()
    assert _moving(lv.shape, lv.stride()) == _moving(v.shape, v.stride())
    assert v.storage_offset() == cell.expected_offset
    if v.numel() > 0:
        assert lv.storage_offset() == cell.expected_offset


# ── Seeded random view chains (depth 1–4) from (4, 6) and (2, 3, 4) ──────────


def _chain_ops(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    numel = math.prod(shape)
    u_dim = rng.randrange(len(shape) + 1)
    ops: list[_View] = [
        lambda t: t.detach(),
        lambda t: t.data,
        lambda t: t.squeeze(),
        lambda t, u=u_dim: t.unsqueeze(u),
        lambda t: t.flatten(),
    ]
    if not shape:
        return ops
    n = len(shape)
    perm = tuple(rng.sample(range(n), n))
    d = rng.randrange(n)
    axis_len = shape[d]
    ops.extend(
        [
            lambda t, p=perm: t.permute(*p),
            lambda t: t.transpose(0, -1),
            lambda t, ax=d: t.movedim(ax, -1),
        ]
    )
    if n >= 2:
        ops.extend([lambda t: t.T, lambda t: t.mT])
        if shape[0] and shape[1]:
            ops.append(lambda t: t.diagonal(0, 0, 1))
    if axis_len >= 2:
        mid = axis_len // 2
        prefix = (slice(None),) * d
        ops.extend(
            [
                lambda t, pf=prefix: t[pf + (0,)],
                lambda t, pf=prefix, m=mid: t[pf + (m,)],
                lambda t, pf=prefix: t[pf + (-1,)],
                lambda t, pf=prefix, m=mid: t[pf + (slice(0, m),)],
                lambda t, pf=prefix, m=mid, al=axis_len: t[pf + (slice(m, al),)],
                lambda t, pf=prefix, al=axis_len: t[pf + (slice(0, al, 2),)],
                lambda t, pf=prefix, al=axis_len: t[pf + (slice(1, al, 2),)],
                lambda t, ax=d, m=mid: t.narrow(ax, 0, m),
                lambda t, ax=d, m=mid, al=axis_len: t.narrow(ax, m, al - m),
                lambda t, ax=d, m=mid: t.unbind(ax)[m],
                lambda t, ax=d, m=mid: t.split(m, dim=ax)[0],
                lambda t, ax=d, m=mid: t.split(m, dim=ax)[-1],
                lambda t, ax=d: t.chunk(2, dim=ax)[0],
                lambda t, ax=d: t.chunk(2, dim=ax)[-1],
                lambda t, ax=d, m=mid: t.tensor_split([m], ax)[1],
                lambda t, ax=d, m=mid: t.unfold(ax, m, 1),
            ]
        )
    if 1 in shape:
        stretch = tuple(2 if s == 1 else -1 for s in shape)
        ops.append(lambda t, st=stretch: t.expand(*st))
    if numel > 0 and numel % 2 == 0:
        ops.append(lambda t, k=numel // 2: t.reshape(2, k))
    return ops


def _try_op(op: _View, t: lucid.Tensor) -> lucid.Tensor | None:
    try:
        return op(t)
    except (RuntimeError, ValueError, IndexError):
        return None


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("root_shape", [(4, 6), (2, 3, 4)])
@pytest.mark.parametrize("seed", range(16))
def test_random_view_chains_match_across_cpu_and_metal(
    root_shape: tuple[int, ...], seed: int
) -> None:
    rng = random.Random(seed)
    for _ in range(6):
        cpu_root = lucid.arange(float(math.prod(root_shape))).reshape(root_shape)
        metal_root = cpu_root.to("metal")
        cpu, metal = cpu_root, metal_root
        keep_alive = [metal_root]
        copied = False
        for _ in range(rng.randrange(1, 5)):
            candidates = _chain_ops(rng, tuple(cpu.shape))
            op = rng.choice(candidates)
            next_cpu = _try_op(op, cpu)
            next_metal = _try_op(op, metal)
            assert (next_cpu is None) == (next_metal is None)
            if next_cpu is None or next_metal is None:
                continue
            if cpu.numel() > 0 and next_cpu.numel() > 0:
                span = cpu_root.numel() * cpu_root.element_size()
                if not 0 <= next_cpu.data_ptr() - cpu_root.data_ptr() < span:
                    copied = True
            cpu, metal = next_cpu, next_metal
            keep_alive.append(metal)
            assert metal.shape == cpu.shape
            assert metal.tolist() == cpu.tolist()
            assert _moving(metal.shape, metal.stride()) == _moving(
                cpu.shape, cpu.stride()
            )
            if not copied and cpu.numel() > 0:
                assert metal.storage_offset() == _cpu_offset(cpu, cpu_root)
                assert (
                    metal.as_strided(metal.shape, metal.stride()).tolist()
                    == metal.tolist()
                )
                assert (
                    metal.as_strided((2,), (1,), 0).tolist()
                    == metal_root.reshape(-1)[:2].tolist()
                )


# ── Additions from LCD-512/513 findings (LCD-517 cards §Additions) ───────────


@pytest.mark.parametrize("device", _DEVICES)
def test_contiguous_returns_self_when_already_contiguous(device: str) -> None:
    x = lucid.arange(12.0, device=device).reshape(3, 4)
    v = x[1:]
    assert v.is_contiguous()
    c = v.contiguous()
    c.add_(10.0)
    assert x[1:].tolist() == [[14.0, 15.0, 16.0, 17.0], [18.0, 19.0, 20.0, 21.0]]


@pytest.mark.parametrize("device", _DEVICES)
def test_is_contiguous_ignores_size_one_axes(device: str) -> None:
    v = lucid.arange(48.0, device=device).reshape(1, 48).t()[:1]
    assert v.shape == (1, 1)
    assert v.is_contiguous() is True


@pytest.mark.parametrize("device", _DEVICES)
def test_as_strided_refuses_negative_storage_offset(device: str) -> None:
    x = lucid.arange(6.0, device=device)
    with pytest.raises(RuntimeError, match="invalid storage offset"):
        x.as_strided((2,), (1,), -1)


@pytest.mark.parametrize("device", _DEVICES)
def test_empty_slice_storage_offset_counts_from_its_start(device: str) -> None:
    x = lucid.arange(24.0, device=device).reshape(4, 6)
    empty = x[:, 3:1]
    assert empty.shape == (4, 0)
    assert empty.storage_offset() == 3
