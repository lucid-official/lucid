"""Which saved tensors share a storage, and the bytes each storage is saved as.

:func:`lucid.save` writes each storage once and each tensor as a view of one
(shape, stride, offset), so tensors that share memory before a save share it
after a load, as views saved together do in the reference.  Neither the
sharing nor the values may be guessed:

* **Sharing is only what can be proven.**  A CPU tensor's elements live in
  memory Lucid owns, so two CPU tensors share exactly when the byte ranges
  they read meet.  A Metal address proves nothing: MLX hands one immutable
  buffer to every value equal to it (``clone()``, ``contiguous()``, a leading
  slice, the weights ``load_state_dict`` copied in), and those tensors never
  see each other's writes.  A Metal tensor is therefore saved as sharing only
  with members of its engine view family (``TensorImpl.family_id``), each
  placed where its layout puts it in the storage.

* **Values are each tensor's own.**  A shared storage is assembled from every
  member's ``to_bytes()``, laid out by its geometry, and each member is then
  read back out of it.  A member whose bytes do not come back is given a
  storage of its own.  Nothing is read from a raw address.

A tensor that shares with no other saved tensor is saved dense: its values
round-trip, and the loaded tensor owns them instead of viewing a buffer that
was mostly not saved.
"""

from typing import final

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor as _T

#: ``id(tensor) -> (shape, stride, offset, storage id)``, in elements.
type _TensorMeta = dict[int, tuple[list[int], list[int], int, int]]
#: ``as_strided``'s window in the storage's elements: size, stride, offset.
type _Window = tuple[tuple[int, ...], tuple[int, ...], int]


def _collect_tensors(obj: object, seen: set[int], out: list[_T]) -> None:
    """Collect each :class:`Tensor` reachable from ``obj`` once, in traversal order.

    Dict keys and values, list / tuple / set members and the ``__dict__`` of
    other objects are walked; a container seen before is not walked again.
    """
    oid = id(obj)
    if oid in seen:
        return
    if isinstance(obj, _T):
        seen.add(oid)
        out.append(obj)
        return
    if isinstance(obj, dict):
        seen.add(oid)
        for k, v in obj.items():
            _collect_tensors(k, seen, out)
            _collect_tensors(v, seen, out)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        seen.add(oid)
        for item in obj:
            _collect_tensors(item, seen, out)
        return
    if not isinstance(obj, type):
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            seen.add(oid)
            _collect_tensors(d, seen, out)


def _contiguous_stride(shape: list[int]) -> list[int]:
    """Row-major element strides of ``shape``.

    The layout ``to_bytes()`` packs a tensor in, and so the layout of every
    storage a tensor is saved alone in.
    """
    stride = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        stride[d] = stride[d + 1] * max(shape[d + 1], 1)
    return stride


def _reach(shape: list[int], stride: list[int]) -> int:
    """Elements from a non-empty geometry's first element to one past its last."""
    return 1 + sum((d - 1) * s for d, s in zip(shape, stride, strict=True))


def _may_overlap(shape: list[int], stride: list[int]) -> bool:
    """Whether the geometry may read an element twice (conservative)."""
    reach = 0
    for s, d in sorted((s, d) for d, s in zip(shape, stride, strict=True) if d > 1):
        if s <= reach:
            return True
        reach += (d - 1) * s
    return False


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _own_geometry(t: _T) -> tuple[list[int], list[int], int]:
    """``t``'s shape, stride and offset in its storage, in elements."""
    return ([int(d) for d in t.shape], [int(s) for s in t.stride()], int(t.storage_offset()))


def _window_positions(size: tuple[int, ...], stride: tuple[int, ...], start: int) -> _T:
    """``start + sum(i * stride)`` over the indices ``i`` of ``size``, on the CPU."""
    import lucid

    at = lucid.full(size, start, dtype=lucid.int64)
    for d, (n, s) in enumerate(zip(size, stride, strict=True)):
        steps = lucid.arange(n, dtype=lucid.int64) * s
        at = at + steps.reshape([n if e == d else 1 for e in range(len(size))])
    return at.reshape(-1)


def _picked(frame: _T, window: _Window) -> _T | None:
    """Which of ``frame``'s elements each of the window's is, by flat index.

    ``None`` when the window reads an element ``frame`` does not hold.
    Found by sorting ``frame``'s positions in the storage: a strided view
    (``y[:, ::2]``, a diagonal) is no dense block, though it may hold every
    element of a window built from its own strides.
    """
    import lucid

    if not frame.numel():
        return None
    f_shape, f_stride, f_offset = _own_geometry(frame)
    shape = tuple(f_shape)
    stride_t = tuple(f_stride)
    size, stride, offset = window
    reach = sum((n - 1) * s for n, s in zip(shape, stride_t, strict=True))
    span = sum((n - 1) * s for n, s in zip(size, stride, strict=True))
    if offset < f_offset or offset + span > f_offset + reach:
        return None
    own = _window_positions(shape, stride_t, f_offset)
    order = lucid.argsort(own)
    ordered = own[order]
    wanted = _window_positions(*window)
    at = lucid.searchsorted(ordered, wanted).clamp(max=ordered.numel() - 1)
    if not bool((ordered[at] == wanted).all().item()):
        return None
    return order[at]


@final
class _Member:
    """A saved tensor and the elements of its address space it reads."""

    __slots__ = ("tensor", "shape", "stride", "first", "raw")

    def __init__(self, tensor: _T) -> None:
        self.tensor = tensor
        self.shape: list[int] = []
        self.stride: list[int] = []
        #: Byte position of the first element in the address space.
        self.first = 0
        self.raw = b""

    def place(self) -> bool:
        """Fix where the member reads; ``False`` if it cannot be placed."""
        shape, stride, offset = _own_geometry(self.tensor)
        if any(s < 0 for s in stride):
            return False
        self.shape = shape
        self.stride = stride
        # A CPU member is placed by address, which a sub-buffer view (a dim-0
        # slice) shares with its base; a Metal member by its offset in the
        # root allocation of its view family.
        if self.tensor.is_metal:
            self.first = offset * self.tensor.element_size()
        else:
            self.first = int(self.tensor.data_ptr())
        return True

    @property
    def end(self) -> int:
        """Byte position one past the last element."""
        return self.first + _reach(self.shape, self.stride) * self.tensor.element_size()


def _address_space(t: _T) -> tuple[str, int]:
    """The space a member's positions are measured in.

    CPU tensors share one space, the process's memory, measured by address.
    A Metal tensor belongs to its engine view family when it has one, or its
    own Python identity, measured from the first element of the family's root
    allocation.  Two Metal tensors that read one MLX buffer are not thereby one
    storage: the buffer of ``ema.load_state_dict(model.state_dict())``'s
    weights is the model's, and the two must not alias after a load.
    """
    if not t.is_metal:
        return ("memory", 0)
    fid = t._impl.family_id()
    return ("family", fid) if fid else ("value", id(t))


@final
class _StoragePlan:
    """The storages ``save`` writes and where each saved tensor sits in one.

    ``meta`` holds the ``(shape, stride, offset, storage id)`` each tensor
    record carries; ``blobs`` the bytes of each storage, by id.
    """

    def __init__(self) -> None:
        self.meta: _TensorMeta = {}
        self.blobs: dict[int, bytes] = {}

    def add_dense(self, t: _T, raw: bytes | None = None) -> None:
        """Give ``t`` a storage of its own, holding its elements row-major."""
        sid = len(self.blobs)
        self.blobs[sid] = bytes(t._impl.to_bytes()) if raw is None else raw
        shape = [int(d) for d in t.shape]
        self.meta[id(t)] = (shape, _contiguous_stride(shape), 0, sid)

    def add_shared(self, members: list[_Member], blob: bytes, lo: int) -> None:
        """One storage ``blob``, starting at byte ``lo``, that ``members`` view."""
        sid = len(self.blobs)
        self.blobs[sid] = blob
        for m in members:
            offset = (m.first - lo) // m.tensor.element_size()
            self.meta[id(m.tensor)] = (m.shape, m.stride, offset, sid)


def _clusters(members: list[_Member]) -> list[list[_Member]]:
    """Members whose byte ranges meet, transitively, in first-byte order."""
    out: list[list[_Member]] = []
    hi = 0
    for m in sorted(members, key=lambda m: (m.first, m.end)):
        if out and m.first < hi:
            out[-1].append(m)
            hi = max(hi, m.end)
        else:
            out.append([m])
            hi = m.end
    return out


def _assemble(cluster: list[_Member], lo: int, n: int) -> _C_engine.TensorImpl:
    """A CPU buffer of ``n`` elements from byte ``lo``, each member written in.

    A member that covers the whole span gives the buffer its bytes — the one
    nearest the storage, if several do.  Otherwise each member is written at
    its geometry, the most derived first, so the least derived wins where two
    disagree; a member that may read an element twice is only read back.
    """
    dtype = cluster[0].tensor._impl.dtype
    itemsize = cluster[0].tensor.element_size()
    cpu = _C_engine.Device.CPU
    covering = [
        m
        for m in cluster
        if m.first == lo and m.stride == _contiguous_stride(m.shape) and _numel(m.shape) == n
    ]
    if covering:
        return _C_engine.TensorImpl.from_bytes(covering[0].raw, [n], dtype, cpu, False)
    buf = _C_engine.zeros([n], dtype, cpu)
    for m in cluster:
        if _may_overlap(m.shape, m.stride):
            continue
        src = _C_engine.TensorImpl.from_bytes(m.raw, m.shape, dtype, cpu, False)
        offset = (m.first - lo) // itemsize
        _C_engine.TensorImpl._make_view(buf, m.shape, m.stride, offset).copy_from(src)
    return buf


def _compact_strided_cluster(plan: _StoragePlan, cluster: list[_Member], n: int) -> bool:
    """Save ``cluster`` in its largest member's packed elements when the span exceeds it."""
    top = max(cluster, key=lambda m: _numel(m.shape))
    if _numel(top.shape) >= n or _may_overlap(top.shape, top.stride):
        return False
    itemsize = top.tensor.element_size()
    remapped: list[tuple[_Member, list[int], int]] = []
    for m in cluster:
        if m is top:
            remapped.append((m, _contiguous_stride(m.shape), 0))
            continue
        window = (tuple(m.shape), tuple(m.stride), m.first // itemsize)
        picked = _picked(top.tensor, window)
        if picked is None:
            return False
        offset = int(picked[0].item())
        step = _contiguous_stride(m.shape)
        stride = [
            int(picked[s].item()) - offset if d > 1 else 1
            for d, s in zip(m.shape, step, strict=True)
        ]
        placed = _window_positions(tuple(m.shape), tuple(stride), offset)
        if not bool((placed == picked).all().item()):
            return False
        remapped.append((m, stride, offset))
    buf = _C_engine.TensorImpl.from_bytes(
        top.raw, [_numel(top.shape)], top.tensor._impl.dtype, _C_engine.Device.CPU, False
    )
    for m, stride, offset in remapped:
        back = _C_engine.TensorImpl._make_view(buf, m.shape, stride, offset)
        if bytes(back.to_bytes()) != m.raw:
            return False
    for m, stride, offset in remapped:
        m.stride = stride
        m.first = offset * itemsize
    k_lo = min(m.first for m in cluster)
    k_hi = max(m.end for m in cluster)
    plan.add_shared(cluster, top.raw[k_lo:k_hi], k_lo)
    return True


def _plan_cluster(plan: _StoragePlan, cluster: list[_Member]) -> None:
    """Save ``cluster`` as one storage, less the members it does not reproduce."""
    itemsize = cluster[0].tensor.element_size()
    lo = cluster[0].first
    n = (max(m.end for m in cluster) - lo) // itemsize
    for m in cluster:
        m.raw = bytes(m.tensor._impl.to_bytes())
    if _compact_strided_cluster(plan, cluster, n):
        return
    buf = _assemble(cluster, lo, n)
    kept: list[_Member] = []
    for m in cluster:
        offset = (m.first - lo) // itemsize
        back = _C_engine.TensorImpl._make_view(buf, m.shape, m.stride, offset)
        if bytes(back.to_bytes()) == m.raw:
            kept.append(m)
        else:
            plan.add_dense(m.tensor, m.raw)
    if len(kept) < 2:
        for m in kept:
            plan.add_dense(m.tensor, m.raw)
        return
    k_lo = min(m.first for m in kept)
    k_hi = max(m.end for m in kept)
    plan.add_shared(kept, bytes(buf.to_bytes())[k_lo - lo : k_hi - lo], k_lo)


def _plan_space(plan: _StoragePlan, members: list[_Member]) -> None:
    """Plan the members that share one address space."""
    placed: list[_Member] = []
    for m in members:
        if m.place():
            placed.append(m)
        else:
            plan.add_dense(m.tensor)
    itemsize = members[0].tensor.element_size()
    for cluster in _clusters(placed):
        aligned = all((m.first - cluster[0].first) % itemsize == 0 for m in cluster)
        if len(cluster) > 1 and aligned:
            _plan_cluster(plan, cluster)
        else:
            for m in cluster:
                plan.add_dense(m.tensor)


def _build_storage_plan(container: object) -> tuple[list[_T], _StoragePlan]:
    """Plan the storages for every tensor reachable from ``container``.

    Returns the tensors the plan refers to by ``id`` — keep them alive until
    the pickle is written — and the plan.
    """
    tensors: list[_T] = []
    _collect_tensors(container, set(), tensors)
    keep_alive: list[_T] = list(tensors)
    plan = _StoragePlan()
    spaces: dict[tuple[str, str, tuple[str, int]], list[_Member]] = {}
    for t in tensors:
        if t.numel() == 0:
            plan.add_dense(t, b"")
            continue
        key = ("metal" if t.is_metal else "cpu", t.dtype._name, _address_space(t))
        spaces.setdefault(key, []).append(_Member(t))
    for members in spaces.values():
        if len(members) == 1:
            plan.add_dense(members[0].tensor)
        else:
            _plan_space(plan, members)
    return keep_alive, plan
