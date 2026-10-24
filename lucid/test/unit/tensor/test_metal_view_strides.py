"""A Metal view's layout is worked out from its op, not replayed on the CPU.

``view`` and ``reshape`` of a traced Metal view — a transpose, a permute, a
slice — need the strides the same view would have on the CPU, to know
whether the new shape can read it in place.  They were read back by
replaying the view on a CPU tensor: a host allocation and a CPU op for every
merge-heads ``attn.permute(0, 2, 1, 3).reshape(b, n, e)``, in every attention
layer, every step (LCD-462).  The engine's view family now keeps each Metal
view's geometry from the op that made it, with where the view starts in its
base's storage, which ``as_strided``, ``stride()`` and ``storage_offset()``
read (LCD-513); these tests hold each rule to the CPU and to zero host work.

Strides are compared on the axes that move an element.  An axis of length
one moves none, and the CPU's choice of stride for it depends on how the
view was composed (a slice squeezes its window axis, which re-strides every
length-one axis), so it says nothing about the layout.  Offsets are compared
against both the CPU view's ``storage_offset()`` and where its first element
sits in its base's buffer (LCD-517).
"""

import itertools
import math
import random
from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_CPU = _C_engine.Device.CPU

type _View = Callable[[lucid.Tensor], lucid.Tensor]


def _moving(shape: tuple[int, ...], stride: tuple[int, ...]) -> tuple[int, ...]:
    """The strides of the axes that move an element; ``()`` when none can."""
    if 0 in shape:
        return ()
    return tuple(st for n, st in zip(shape, stride, strict=True) if n != 1)


def _contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """The reference's contiguity: row-major steps on every axis that moves.

    Matches ``is_contiguous()`` on both devices, ignoring length-one axes.
    """
    if 0 in shape:
        return True
    expected = 1
    for n, st in reversed(list(zip(shape, stride, strict=True))):
        if n != 1:
            if st != expected:
                return False
            expected *= n
    return True


def _cpu_offset(view: lucid.Tensor, root: lucid.Tensor) -> int:
    """Where a CPU view's first element sits in ``root``'s buffer, in elements."""
    return (view.data_ptr() - root.data_ptr()) // view.element_size()


def _base(device: str) -> lucid.Tensor:
    # A traced view itself, so the rule under test reads its input's strides
    # from another rule.
    return lucid.arange(120.0).reshape(2, 3, 4, 5).to(device).permute(3, 1, 0, 2)


#: One view per rule, and per spelling of the ones a free function shares.
#: Every one is taken of ``_base`` — shape (5, 3, 2, 4).
VIEWS: dict[str, _View] = {
    "transpose": lambda t: t.transpose(0, 2),
    "transpose-kwargs": lambda t: t.transpose(dim0=-1, dim1=1),
    "transpose-all": lambda t: t.transpose(),
    "lucid.transpose": lambda t: lucid.transpose(t, 0, 2),
    "swapaxes": lambda t: t.swapaxes(1, -1),
    "permute-spread": lambda t: t.permute(2, 0, 3, 1),
    "permute-tuple": lambda t: t.permute((2, 0, 3, 1)),
    "permute-kwargs": lambda t: t.permute(dims=[2, 0, 3, 1]),
    "lucid.permute": lambda t: lucid.permute(t, (2, 0, 3, -3)),
    "movedim": lambda t: t.movedim(0, -1),
    "movedim-lists": lambda t: t.movedim((0, 1), (3, 2)),
    "lucid.movedim": lambda t: lucid.movedim(t, source=[1], destination=[0]),
    "t": lambda t: t[0, 0].t(),
    "T": lambda t: t.T,
    "mT": lambda t: t.mT,
    "narrow": lambda t: t.narrow(0, 1, 3),
    "diagonal": lambda t: t.diagonal(),
    "diagonal-dims": lambda t: t.diagonal(1, 0, 2),
    "lucid.diagonal": lambda t: lucid.diagonal(t, offset=-1, dim1=3, dim2=0),
    "unfold": lambda t: t.unfold(0, 2, 2),
    "unfold-last": lambda t: t.unfold(-1, 3, 1),
    "expand-leading": lambda t: t.expand(2, 5, 3, 2, 4),
    "expand-stretch": lambda t: t[:, :1].expand(-1, 3, -1, -1),
    "broadcast_to": lambda t: t[:1].broadcast_to((3, 5, 3, 2, 4)),
    "expand_as": lambda t: t[:, :, :1].expand_as(t),
    "as_strided": lambda t: t.as_strided((2, 3), (4, 1)),
    "as_strided-offset": lambda t: t[1:].as_strided((2, 3), (4, 1), 7),
    "lucid.as_strided": lambda t: lucid.as_strided(t.t(), (3,), (5,)),
    "split": lambda t: t.split(2, dim=0)[1],
    "lucid.split": lambda t: lucid.split(t, [1, 3], dim=3)[1],
    "chunk": lambda t: t.chunk(2, dim=3)[1],
    "unbind": lambda t: t.unbind(1)[2],
    "lucid.unbind": lambda t: lucid.unbind(t, dim=-1)[0],
    "index-int": lambda t: t[1],
    "index-step": lambda t: t[:, ::2],
    "index-mixed": lambda t: t[1:, 0, ::3],
    "index-none": lambda t: t[1:, None, ..., 2],
    "index-none-inner": lambda t: t[..., None],
    "index-empty": lambda t: t[3:1],
}


@pytest.mark.parametrize("make", list(VIEWS.values()), ids=list(VIEWS))
def test_a_metal_view_layout_dispatches_nothing_and_is_the_cpus(make: _View) -> None:
    view = make(_base("metal"))
    host = _C_engine.memory_stats(_CPU).alloc_count
    with lucid.profiler.profile() as prof:
        reported = (view.stride(), view.storage_offset(), view.is_contiguous())
    assert [event.name for event in prof.events()] == []
    assert _C_engine.memory_stats(_CPU).alloc_count == host
    assert reported == (
        view.stride(),
        view.storage_offset(),
        _contiguous(view.shape, view.stride()),
    )

    base = _base("cpu")
    cpu = make(base)
    assert view.shape == cpu.shape
    assert _moving(view.shape, view.stride()) == _moving(cpu.shape, cpu.stride())
    assert (
        view.is_contiguous()
        == cpu.is_contiguous()
        == _contiguous(cpu.shape, cpu.stride())
    )
    assert view.storage_offset() == cpu.storage_offset()
    if cpu.numel():
        assert view.storage_offset() == _cpu_offset(cpu, base)


def test_merging_heads_on_metal_runs_no_cpu_op() -> None:
    # MultiheadAttention's merge: the step-cost budget's transformer cell.
    x = lucid.randn(2, 8, 32, device="metal")
    heads = x.reshape(2, 8, 4, 8).permute(0, 2, 1, 3)
    host = _C_engine.memory_stats(_CPU).alloc_count
    with lucid.profiler.profile() as prof:
        merged = heads.permute(0, 2, 1, 3).reshape(2, 8, 32)
    assert [event.name for event in prof.events()] == ["permute", "reshape"]
    assert _C_engine.memory_stats(_CPU).alloc_count == host
    assert merged.tolist() == x.tolist()


def test_a_reversed_slice_is_a_tensor_of_its_own() -> None:
    # The CPU gathers ``x[::-1]`` into a copy (LCD-512), so on Metal it is no
    # view of ``x`` either: ``view`` steps through the copy's own layout.
    x = lucid.arange(6.0, device="metal")
    reversed_ = x[::-1]
    assert reversed_._impl.family_id() == 0
    assert reversed_.view(2, 3).tolist() == [[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]]
    assert x[1:4:-1]._impl.family_id() == 0


# ── Random chains, held to the CPU ───────────────────────────────────────────


def _shape_of(rng: random.Random, numel: int) -> tuple[int, ...]:
    """A random shape with ``numel`` elements, length-one axes included."""
    dims: list[int] = []
    left = numel
    while left > 1 and len(dims) < 4:
        size = rng.choice([d for d in range(2, left + 1) if left % d == 0])
        dims.append(size)
        left //= size
    if left > 1:
        dims.append(left)
    for _ in range(rng.randrange(2)):
        dims.insert(rng.randrange(len(dims) + 1), 1)
    rng.shuffle(dims)
    return tuple(dims)


def _key(rng: random.Random, shape: tuple[int, ...]) -> tuple[object, ...]:
    parts: list[object] = []
    for n in shape[: rng.randrange(len(shape) + 1)]:
        pick = rng.randrange(5)
        if pick == 0 and n:
            parts.append(rng.randrange(-n, n))
        elif pick == 1:
            start = rng.randrange(n + 1)
            parts.append(slice(start, rng.randrange(start, n + 1), rng.randrange(1, 4)))
        elif pick == 2:
            # A copy on the CPU (LCD-512): what follows is a view of the copy.
            parts.append(slice(rng.choice([None, n - 1]), None, rng.choice([-1, -2])))
        else:
            parts.append(slice(None))
        if rng.randrange(5) == 0:
            parts.append(None)
    if rng.randrange(3) == 0:
        parts.append(Ellipsis)
    return tuple(parts)


def _windows(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    """``as_strided`` built from the view's own ``stride()``, as callers build it."""
    i = rng.randrange(len(shape))
    width = rng.randrange(1, shape[i] + 1) if shape[i] else 0
    part = tuple(rng.randrange(n + 1) for n in shape)
    return [
        lambda t: t.as_strided(t.shape, t.stride()),
        lambda t: t.as_strided(part, t.stride()),
        # Sliding windows along axis ``i``, from the view's first element.
        lambda t: t.as_strided(
            (shape[i] - width + 1, width), (t.stride(i), t.stride(i))
        ),
    ]


def _ranked_views(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    n = len(shape)
    perm = rng.sample(range(n), n)
    i, j = rng.randrange(n), rng.randrange(n)
    size = rng.randrange(1, shape[i] + 1) if shape[i] else 0
    start = rng.randrange(shape[i] - size + 1) if shape[i] else 0
    step = rng.randrange(1, 4)
    piece = rng.randrange(max(shape[i], 1))
    chunks = rng.randrange(1, 4)
    stretch = tuple(rng.choice([2, 3]) if s == 1 else -1 for s in shape)
    views: list[_View] = [
        lambda t: t.permute(*perm),
        lambda t: t.transpose(i, j),
        lambda t: t.movedim(i, j),
        lambda t: t.narrow(i, start, size),
        lambda t: t.unbind(i)[piece] if shape[i] else t,
        lambda t: t.split(max(size, 1), dim=i)[0],
        lambda t: t.chunk(chunks, dim=i)[-1] if shape[i] else t,
        lambda t: t.expand(*stretch),
        lambda t: t.expand(2, *shape),
        lambda t: t.T,
    ]
    if 0 not in shape:
        # An empty slice starts at its first index in the reference and at 0 on
        # the CPU; a window from there says nothing either of them reads.
        views += _windows(rng, shape)
    if shape[i]:
        views.append(lambda t: t.unfold(i, size, step))
    if n >= 2 and i != j:
        offset = rng.randrange(-2, 3)
        views += [lambda t: t.diagonal(offset, i, j), lambda t: t.mT]
    return views


def _views(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    """Every view op the random chains draw from, with arguments for ``shape``."""
    numel = 1
    for n in shape:
        numel *= n
    target = _shape_of(rng, numel) if numel else (0, *shape)
    key = _key(rng, shape)
    views: list[_View] = [
        lambda t: t[key],
        lambda t: t.reshape(*target),
        lambda t: t.unsqueeze(rng.randrange(len(shape) + 1)),
        lambda t: t.squeeze(),
        lambda t: t.detach(),
    ]
    if shape:
        views += _ranked_views(rng, shape)
    return views


class _Chain:
    """One random chain, on both devices, with the buffer the CPU view reads."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.cpu = lucid.arange(float(math.prod(shape))).reshape(shape)
        self.metal = self.cpu.to("metal")
        # The tensors that own each device's storage, held so a Metal window
        # may read any of it, as a CPU one may; ``None`` once the view reads
        # a copy only the CPU can see the start of.
        self.cpu_root: lucid.Tensor | None = self.cpu
        self.metal_root: lucid.Tensor | None = self.metal
        self.steps = [f"{shape}"]

    def where(self) -> str:
        return " ".join(self.steps)

    def step(self, cpu: lucid.Tensor, metal: lucid.Tensor) -> None:
        root = self.cpu_root
        if root is not None and cpu.numel():
            span = root.numel() * root.element_size()
            if not 0 <= cpu.data_ptr() - root.data_ptr() < span:
                # The CPU copied (a slice stepping backwards): the copy is a
                # storage of its own on both devices.  A key that goes on
                # past the slice views the copy from somewhere inside it.
                if self.metal_root is not None:
                    root_fid = self.metal_root._impl.family_id()
                    assert (
                        root_fid == 0 or metal._impl.family_id() != root_fid
                    ), self.where()
                copied = cpu.storage_offset() == 0 and cpu.is_contiguous()
                self.cpu_root, self.metal_root = (
                    (cpu, metal) if copied else (None, None)
                )
        self.cpu, self.metal = cpu, metal
        self.steps.append(f"-> {tuple(cpu.shape)}")
        _same_view(self.cpu, self.metal, self.cpu_root, self.where())


def _same_view(
    cpu: lucid.Tensor, metal: lucid.Tensor, root: lucid.Tensor | None, where: str
) -> None:
    assert metal.shape == cpu.shape, where
    assert _moving(metal.shape, metal.stride()) == _moving(
        cpu.shape, cpu.stride()
    ), where
    if cpu.numel():
        assert metal.is_contiguous() == cpu.is_contiguous(), where
        if root is not None:
            assert (
                metal.storage_offset() == cpu.storage_offset() == _cpu_offset(cpu, root)
            ), where
    assert metal.tolist() == cpu.tolist(), where


def _attempt(view: _View, t: lucid.Tensor) -> lucid.Tensor | None:
    try:
        return view(t)
    except RuntimeError:
        return None


def _a_window_reads_the_storage(
    rng: random.Random, chain: _Chain, root: lucid.Tensor
) -> None:
    """A window anywhere in the storage reads what the storage holds there.

    Both CPU and Metal ``as_strided`` count offsets in the root allocation
    frame (LCD-517).
    """
    flat = root.reshape(-1).tolist()
    size = tuple(rng.randrange(1, 4) for _ in range(rng.randrange(1, 3)))
    stride = tuple(rng.randrange(4) for _ in size)
    reach = sum((n - 1) * s for n, s in zip(size, stride, strict=True))
    if reach >= len(flat):
        return
    offset = rng.randrange(len(flat) - reach)
    want = lucid.tensor(
        [
            flat[offset + sum(i * s for i, s in zip(at, stride, strict=True))]
            for at in itertools.product(*(range(n) for n in size))
        ]
    ).reshape(size)
    got_cpu = chain.cpu.as_strided(size, stride, offset)
    got = chain.metal.as_strided(size, stride, offset)
    assert got_cpu.tolist() == want.tolist(), (chain.where(), size, stride, offset)
    assert got.tolist() == want.tolist(), (chain.where(), size, stride, offset)


@pytest.mark.parametrize("seed", range(24))
def test_random_view_chains_have_the_cpus_layout(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(12):
        chain = _Chain(_shape_of(rng, rng.choice([1, 6, 8, 12, 24, 36, 48])))
        for _ in range(rng.randrange(1, 5)):
            view = rng.choice(_views(rng, tuple(chain.cpu.shape)))
            # Draw once and replay the same op on both devices.
            state = rng.getstate()
            next_cpu = _attempt(view, chain.cpu)
            rng.setstate(state)
            next_metal = _attempt(view, chain.metal)
            assert (next_cpu is None) == (next_metal is None), chain.where()
            if next_cpu is not None and next_metal is not None:
                chain.step(next_cpu, next_metal)
        # An empty view's address says nothing of its storage.
        if chain.cpu_root is not None and chain.cpu.numel():
            _a_window_reads_the_storage(rng, chain, chain.cpu_root)
        # ``view`` refuses on Metal exactly where it refuses on the CPU.
        cpu, metal = chain.cpu, chain.metal
        target = _shape_of(rng, cpu.numel()) if cpu.numel() else (0,)
        cpu_view = _attempt(lambda t: t.view(*target), cpu)
        metal_view = _attempt(lambda t: t.view(*target), metal)
        assert (cpu_view is None) == (metal_view is None), (chain.where(), target)
        if cpu_view is not None and metal_view is not None:
            chain.step(cpu_view, metal_view)


# ── The Python chunk rule is the engine's ────────────────────────────────────


def _random_layout(rng: random.Random, device: str = "cpu") -> lucid.Tensor:
    """A view of random shape and strides over a buffer large enough for it."""
    shape = [rng.choice([1, 1, 2, 3, 4]) for _ in range(rng.randrange(5))]
    if shape and rng.randrange(8) == 0:
        shape[rng.randrange(len(shape))] = 0
    stride = [rng.choice([0, 1, 2, 3, 4, 6, 8, 12, 24]) for _ in shape]
    span = 1 + sum((n - 1) * st for n, st in zip(shape, stride, strict=True) if n)
    buffer = lucid.zeros(span, device=device)
    return _wrap(_C_engine.TensorImpl._make_view(buffer._impl, shape, stride, 0))


@pytest.mark.parametrize("seed", range(16))
def test_the_python_view_rule_is_the_engines(seed: int) -> None:
    rng = random.Random(1000 + seed)
    for _ in range(40):
        state = rng.getstate()
        a_cpu = _random_layout(rng, "cpu")
        rng.setstate(state)
        a_metal = _random_layout(rng, "metal")
        shape = tuple(a_cpu.shape)
        target = _shape_of(rng, a_cpu.numel()) if a_cpu.numel() else (0, *shape)
        try:
            cpu_stride: tuple[int, ...] | None = tuple(a_cpu.view(*target).stride())
        except RuntimeError:
            cpu_stride = None
        try:
            metal_stride: tuple[int, ...] | None = tuple(a_metal.view(*target).stride())
        except RuntimeError:
            metal_stride = None
        assert (metal_stride is None) == (cpu_stride is None), (
            shape,
            a_cpu.stride(),
            target,
        )
        if metal_stride is not None and cpu_stride is not None:
            assert _moving(target, metal_stride) == _moving(target, cpu_stride)
        assert a_metal.squeeze().stride() == a_cpu.squeeze().stride()
        axis = rng.randrange(len(shape) + 1)
        assert a_metal.unsqueeze(axis).stride() == a_cpu.unsqueeze(axis).stride()
