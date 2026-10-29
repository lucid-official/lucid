"""A checkpoint is what was saved: tensors, tied weights and training state.

The serialization files pin specific bugs (views, SafeTensors dtypes, shards).
This module asks the generic question over the whole grid a user's checkpoint
can contain, and stops at what a user can observe after ``load``:

* **Tensors** — every dtype the device holds (bfloat16, float16, complex64,
  int64, bool among them) in every layout (dense, transposed, strided, a column,
  0-d, empty), on CPU and Metal, through each of ``lucid.save``,
  ``save_sharded`` and ``save_safetensors``.  Dtype, device, shape and the
  elements' *bits* — ``-0.0``, NaN, the infinities and subnormals included —
  come back as they went in, and ``map_location`` moves them across devices
  without changing any of that.
* **Tied weights** — one ``Parameter`` in two modules is one tensor in the
  state_dict's storage, in the file once, and one parameter again after
  ``load_state_dict``; the reference agrees.
* **Training state** — model, optimizer, LR scheduler and ``GradScaler`` saved
  mid-run and restored into fresh objects resume the run exactly (CPU: bit for
  bit; Metal: to GPU round-off), for every optimizer, with a tied weight and
  batch-norm buffers in the model.

Isolated views are written densely (their layout is a storage detail), so the
layout after a load is only held to the logical elements — except when the view
is saved with its base, where the strides must match the reference's.
"""

import io
import pickle
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.amp import GradScaler
from lucid.optim import lr_scheduler as sched
from lucid.serialization import (
    load,
    load_safetensors,
    load_sharded,
    save,
    save_safetensors,
    save_sharded,
)
from lucid.test._fixtures.devices import device_dtype_params, device_supports, metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

_DTYPES = [
    lucid.float32,
    lucid.float64,
    lucid.float16,
    lucid.bfloat16,
    lucid.complex64,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool,
]
_FLOATING = {lucid.float32, lucid.float64, lucid.float16, lucid.bfloat16}

# Elements an arithmetic round trip would launder.
_SPECIAL = [-0.0, float("nan"), float("inf"), float("-inf"), 1e-7, 65504.0, -1.5, 0.0]


# ── tensors ──────────────────────────────────────────────────────────────────


def _dense(dtype: lucid.dtype, device: str) -> lucid.Tensor:
    """A (4, 6) tensor whose elements are all distinct bit patterns of ``dtype``."""
    flat = lucid.arange(24, device="cpu").to(lucid.float32)
    if dtype in _FLOATING:
        for i, v in enumerate(_SPECIAL):
            flat[3 * i] = v
    elif dtype == lucid.bool:
        flat = (flat % 3 == 0).to(lucid.float32)
    return flat.reshape(4, 6).to(dtype).to(device)


_LAYOUTS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "dense": lambda x: x,
    "transpose": lambda x: x.t(),
    "step": lambda x: x[::2, 1::2],
    "column": lambda x: x[:, 2],
    "scalar": lambda x: x[1, 2].reshape(()),
    "empty_rows": lambda x: x[:0],
    "empty_3d": lambda x: x[:2, :0].reshape(2, 0, 1),
}


def _bits(t: lucid.Tensor) -> bytes:
    """The elements in logical row-major order, as raw bytes — NaN-safe."""
    return bytes(t.to("cpu").contiguous()._impl.to_bytes())


def _pickle_roundtrip(obj: object, **kwargs: object) -> object:
    buf = io.BytesIO()
    save(obj, buf)
    buf.seek(0)
    return load(buf, **kwargs)  # type: ignore[arg-type]


def _sharded_roundtrip(obj: object, *, shard_mb: float = 1e-6, **kwargs: object) -> object:
    # The default limit puts every tensor in a shard of its own.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lucid-ckpt-") as d:
        save_sharded(obj, Path(d) / "ck", shard_size_mb=shard_mb)
        return load_sharded(Path(d) / "ck", **kwargs)  # type: ignore[arg-type]


def _one_shard_roundtrip(obj: object) -> object:
    return _sharded_roundtrip(obj, shard_mb=1024.0)


def _safetensors_roundtrip(obj: dict[str, lucid.Tensor], device: str = "cpu") -> object:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lucid-ckpt-") as d:
        path = str(Path(d) / "ck.safetensors")
        save_safetensors(obj, path)
        return load_safetensors(path, device=device)


_FORMATS: dict[str, Callable[..., object]] = {
    "pickle": _pickle_roundtrip,
    "sharded": _sharded_roundtrip,
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("fmt", list(_FORMATS))
@pytest.mark.parametrize("device, dtype", device_dtype_params(_DTYPES))
def test_a_tensor_comes_back_as_it_went_in(
    device: str, dtype: lucid.dtype, fmt: str, layout: str
) -> None:
    t = _LAYOUTS[layout](_dense(dtype, device))
    got = _FORMATS[fmt]({"t": t})["t"]  # type: ignore[index]

    assert got.dtype == dtype and got.device.type == device
    assert tuple(got.shape) == tuple(t.shape)
    assert _bits(got) == _bits(t)
    assert got.requires_grad is False


@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("device, dtype", device_dtype_params(_DTYPES))
def test_a_safetensors_tensor_comes_back_as_it_went_in(
    device: str, dtype: lucid.dtype, layout: str
) -> None:
    t = _LAYOUTS[layout](_dense(dtype, device))
    got = _safetensors_roundtrip({"t": t}, device=device)["t"]  # type: ignore[index]

    assert got.dtype == dtype and got.device.type == device
    assert tuple(got.shape) == tuple(t.shape)
    assert _bits(got) == _bits(t)


@pytest.mark.parametrize("layout", ["dense", "transpose", "scalar", "empty_rows"])
@pytest.mark.parametrize("dst", _DEVICES)
@pytest.mark.parametrize("device, dtype", device_dtype_params(_DTYPES))
def test_map_location_moves_a_tensor_and_keeps_everything_else(
    device: str, dtype: lucid.dtype, dst: str, layout: str
) -> None:
    if not device_supports(dst, dtype):
        pytest.skip(f"{dst} cannot hold {dtype}")
    t = _LAYOUTS[layout](_dense(dtype, device))
    for fmt in (_pickle_roundtrip, _sharded_roundtrip):
        got = fmt({"t": t}, map_location=dst)["t"]  # type: ignore[index]
        assert got.device.type == dst and got.dtype == dtype
        assert tuple(got.shape) == tuple(t.shape)
        assert _bits(got) == _bits(t)


@pytest.mark.parametrize("device", _DEVICES)
def test_a_saved_tensor_is_a_snapshot(device: str) -> None:
    # Bytes are taken at ``save``: writing the tensor afterwards, or loading
    # the file twice, never reaches another copy.
    t = lucid.arange(6, device=device).to(lucid.float32)
    buf = io.BytesIO()
    save({"t": t}, buf)
    with lucid.no_grad():
        t.add_(100.0)
    buf.seek(0)
    a = load(buf)["t"]
    buf.seek(0)
    b = load(buf)["t"]
    assert a.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    with lucid.no_grad():
        a.add_(1.0)
    assert b.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize("device", _DEVICES)
def test_an_isolated_view_keeps_its_elements_in_whatever_layout(device: str) -> None:
    # The layout of a view saved alone is the saver's choice (it writes the
    # elements densely); the elements, shape and dtype are not.
    base = _dense(lucid.float32, device)
    for name, make in _LAYOUTS.items():
        v = make(base)
        got = _pickle_roundtrip({"v": v})["v"]  # type: ignore[index]
        assert _bits(got) == _bits(v), name
        assert got.numel() == v.numel() and got.ndim == v.ndim, name


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", ["transpose", "step", "column", "dense"])
def test_a_view_saved_with_its_base_keeps_the_references_strides(
    ref: ModuleType, device: str, name: str
) -> None:
    base = _dense(lucid.float32, device)
    got = _pickle_roundtrip({"base": base, "view": _LAYOUTS[name](base)})
    assert isinstance(got, dict)

    rbase = ref.arange(24, dtype=ref.float32).reshape(4, 6)
    rviews = {
        "transpose": rbase.t(),
        "step": rbase[::2, 1::2],
        "column": rbase[:, 2],
        "dense": rbase,
    }
    buf = io.BytesIO()
    ref.save({"base": rbase, "view": rviews[name]}, buf)
    buf.seek(0)
    want = ref.load(buf)

    assert got["view"].stride() == tuple(want["view"].stride())
    assert got["view"].is_contiguous() == want["view"].is_contiguous()
    assert got["base"].stride() == tuple(want["base"].stride())


@pytest.mark.parametrize("device", _DEVICES)
def test_parameters_and_flags_survive_in_every_container(device: str) -> None:
    p = nn.Parameter(lucid.randn(3, device=device))
    frozen = nn.Parameter(lucid.randn(2, device=device), requires_grad=False)
    leaf = lucid.randn(2, device=device, requires_grad=True)
    nested = {"a": [p, (frozen, {"b": leaf})], "n": 3, "s": "x", "f": 0.5, "none": None}
    for fmt in (_pickle_roundtrip, _sharded_roundtrip):
        got = fmt(nested)
        assert isinstance(got, dict)
        gp, (gf, inner) = got["a"]
        assert type(gp) is nn.Parameter and gp.requires_grad
        assert type(gf) is nn.Parameter and not gf.requires_grad
        assert type(inner["b"]) is lucid.Tensor and inner["b"].requires_grad
        assert (got["n"], got["s"], got["f"], got["none"]) == (3, "x", 0.5, None)
        assert _bits(gp) == _bits(p) and _bits(gf) == _bits(frozen)


def test_a_file_path_and_a_file_object_are_the_same_checkpoint(tmp_path: Path) -> None:
    sd = {"w": _dense(lucid.bfloat16, "cpu"), "i": _dense(lucid.int64, "cpu")}
    save(sd, str(tmp_path / "a.pt"))
    save(sd, tmp_path / "b.pt")
    with open(tmp_path / "c.pt", "wb") as fh:
        save(sd, fh)
    want = {k: _bits(v) for k, v in sd.items()}
    for source in (str(tmp_path / "a.pt"), tmp_path / "b.pt"):
        assert {k: _bits(v) for k, v in load(source).items()} == want
    with open(tmp_path / "c.pt", "rb") as fh:
        assert {k: _bits(v) for k, v in load(fh).items()} == want


# ── tied weights ─────────────────────────────────────────────────────────────


class _Tied(nn.Module):
    """Two linears sharing one weight, a norm with buffers, and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(6, 6)
        self.b = nn.Linear(6, 6)
        self.b.weight = self.a.weight
        self.norm = nn.BatchNorm1d(6)
        self.head = nn.Linear(6, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.norm(self.b(self.a(x).tanh())).relu())


_TIE_FORMATS: dict[str, Callable[..., object]] = {
    "pickle": _pickle_roundtrip,
    "one_shard": _one_shard_roundtrip,
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fmt", list(_TIE_FORMATS))
def test_a_tied_weight_is_saved_once_and_shared_after_load(device: str, fmt: str) -> None:
    lucid.manual_seed(0)
    m = _Tied().to(device)
    sd = m.state_dict()
    assert list(sd)[:4] == ["a.weight", "a.bias", "b.weight", "b.bias"]

    got = _TIE_FORMATS[fmt](sd)
    assert isinstance(got, dict)
    # One storage: a write through either name is read through the other.
    with lucid.no_grad():
        got["a.weight"][0, 0] = 77.0
    assert got["b.weight"][0, 0].item() == 77.0
    assert got["a.bias"][0].item() != 77.0

    # And into a module: the tie is the module's, and it holds the loaded values.
    fresh = _Tied().to(device)
    fresh.load_state_dict(got)
    assert fresh.a.weight is fresh.b.weight
    assert fresh.a.weight[0, 0].item() == 77.0


@pytest.mark.xfail(
    strict=True,
    reason="save_sharded packs keys into shards by size alone: a tied weight whose names fall "
    "in different shards is written once per shard and loads as unrelated tensors "
    "(FOUND by TRACK-C; pack tensors that share a storage into one shard)",
)
def test_a_tied_weight_across_a_shard_boundary_is_saved_once_and_shared() -> None:
    w = nn.Parameter(lucid.randn(64, 64))  # 16 KiB
    fillers = {f"f{i}": lucid.randn(64, 64) for i in range(3)}
    one_shard, tied, untied = (
        _sharded_size({"emb": w, **fillers, "head": lucid.randn(64, 64)}, 1024.0),
        _sharded_size({"emb": w, **fillers, "head": w}, 0.03),
        _sharded_size({"emb": w, **fillers, "head": lucid.randn(64, 64)}, 0.03),
    )
    assert one_shard > 0
    assert tied < untied, "the tied weight is written once, not once per shard"
    got = _sharded_roundtrip({"emb": w, **fillers, "head": w}, shard_mb=0.03)
    assert isinstance(got, dict)
    with lucid.no_grad():
        got["emb"][0, 0] = 77.0
    assert got["head"][0, 0].item() == 77.0


def _sharded_size(obj: object, shard_mb: float) -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lucid-ckpt-") as d:
        save_sharded(obj, Path(d) / "ck", shard_size_mb=shard_mb)
        return sum(f.stat().st_size for f in Path(d).rglob("*.lucid"))


@pytest.mark.parametrize("device", _DEVICES)
def test_a_tied_weight_does_not_double_the_file(device: str) -> None:
    w = nn.Parameter(lucid.randn(128, 128, device=device))
    one, two = io.BytesIO(), io.BytesIO()
    save({"a": w}, one)
    save({"a": w, "b": w}, two)
    assert len(two.getvalue()) < len(one.getvalue()) + 512


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.filterwarnings("ignore:lucid.load.. with weights_only=False:UserWarning")
def test_a_whole_module_keeps_its_ties(device: str) -> None:
    m = _Tied().to(device)
    got = _pickle_roundtrip(m, weights_only=False)
    assert isinstance(got, _Tied)
    assert got.a.weight is got.b.weight and type(got.a.weight) is nn.Parameter
    assert got.a.weight.requires_grad
    x = lucid.randn(5, 6, device=device)
    np.testing.assert_array_equal(m.eval()(x).numpy(), got.eval()(x).numpy())


@pytest.mark.parity
def test_tied_weights_load_the_way_the_reference_loads_them(ref: ModuleType) -> None:
    class RefTied(ref.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.a = ref.nn.Linear(4, 4)
            self.b = ref.nn.Linear(4, 4)
            self.b.weight = self.a.weight

    rm = RefTied()
    buf = io.BytesIO()
    ref.save(rm.state_dict(), buf)
    buf.seek(0)
    rsd = ref.load(buf)
    ref_shares = rsd["a.weight"].data_ptr() == rsd["b.weight"].data_ptr()

    class LucidTied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.b.weight = self.a.weight

    sd = _pickle_roundtrip(LucidTied().state_dict())
    assert isinstance(sd, dict)
    with lucid.no_grad():
        sd["a.weight"][0, 0] = 5.0
    assert (sd["b.weight"][0, 0].item() == 5.0) is ref_shares is True


# ── optimizer / scheduler / scaler state: resume ─────────────────────────────

_OPTS: dict[str, Callable[[list[dict[str, object]]], optim.Optimizer]] = {
    "SGD": lambda g: optim.SGD(g, lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-3),
    "Adam": lambda g: optim.Adam(g, lr=0.01, amsgrad=True),
    "AdamW": lambda g: optim.AdamW(g, lr=0.01),
    "RMSprop": lambda g: optim.RMSprop(g, lr=0.01, momentum=0.5, centered=True),
    "Adagrad": lambda g: optim.Adagrad(g, lr=0.05),
    "Adadelta": lambda g: optim.Adadelta(g),
    "Adamax": lambda g: optim.Adamax(g, lr=0.01),
    "NAdam": lambda g: optim.NAdam(g, lr=0.01),
    "RAdam": lambda g: optim.RAdam(g, lr=0.01),
    "ASGD": lambda g: optim.ASGD(g, lr=0.01),
}


class _Run:
    """A model, an optimizer over two parameter groups, a scheduler and a scaler."""

    def __init__(self, device: str, dtype: lucid.dtype, opt: str, scheduled: bool) -> None:
        lucid.manual_seed(3)
        self.device, self.dtype = device, dtype
        self.model = _Tied().to(device).to(dtype)
        m = self.model
        groups: list[dict[str, object]] = [
            {"params": list(m.a.parameters()) + list(m.b.parameters()), "lr": 0.02},
            {"params": list(m.norm.parameters()) + list(m.head.parameters())},
        ]
        self.opt = _OPTS[opt](groups)
        self.sched = sched.CosineAnnealingLR(self.opt, T_max=10) if scheduled else None
        self.scaler = GradScaler(init_scale=2.0**8, growth_interval=2)

    def step(self, k: int) -> float:
        lucid.manual_seed(50 + k)
        x = lucid.randn(10, 6, device=self.device).to(self.dtype)
        y = lucid.randn(10, 3, device=self.device).to(self.dtype)
        self.opt.zero_grad()
        loss = ((self.model(x) - y) ** 2).mean()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.sched is not None:
            self.sched.step()
        return float(loss.item())

    def checkpoint(self) -> object:
        return _pickle_roundtrip(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "sched": None if self.sched is None else self.sched.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
        )

    def restore(self, ck: dict[str, object]) -> None:
        self.model.load_state_dict(ck["model"])  # type: ignore[arg-type]
        self.opt.load_state_dict(ck["opt"])  # type: ignore[arg-type]
        if self.sched is not None:
            self.sched.load_state_dict(ck["sched"])  # type: ignore[arg-type]
        self.scaler.load_state_dict(ck["scaler"])  # type: ignore[arg-type]

    def params(self) -> list[np.ndarray]:
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        return [t.detach().to(lucid.float32).numpy().copy() for t in tensors]


def _resume_cells() -> list:
    cells = []
    for device in _DEVICES:
        for dtype, tag in ((lucid.float32, "fp32"), (lucid.bfloat16, "bf16")):
            for opt in _OPTS:
                for scheduled in (False, True):
                    cells.append(
                        pytest.param(
                            device, dtype, opt, scheduled,
                            id=f"{device}-{tag}-{opt}-{'cosine' if scheduled else 'const'}",
                        )
                    )  # fmt: skip
    return cells


@pytest.mark.parametrize("device, dtype, opt, scheduled", _resume_cells())
def test_a_run_resumed_from_a_checkpoint_continues_exactly(
    device: str, dtype: lucid.dtype, opt: str, scheduled: bool
) -> None:
    live = _Run(device, dtype, opt, scheduled)
    for k in range(3):
        live.step(k)
    ck = live.checkpoint()
    assert isinstance(ck, dict)
    tail = [live.step(k) for k in range(3, 7)]

    resumed = _Run(device, dtype, opt, scheduled)
    resumed.restore(ck)
    tail_resumed = [resumed.step(k) for k in range(3, 7)]

    # Same device, same program: CPU is deterministic to the bit; a GPU run
    # may differ in the last place from one launch to the next.
    tol = {"rtol": 0.0, "atol": 0.0} if device == "cpu" else {"rtol": 1e-5, "atol": 1e-6}
    np.testing.assert_allclose(tail_resumed, tail, **tol)
    for a, b in zip(resumed.params(), live.params(), strict=True):
        np.testing.assert_allclose(a, b, **tol)
    assert resumed.scaler.get_scale() == live.scaler.get_scale()
    if scheduled:
        assert resumed.opt.param_groups[0]["lr"] == live.opt.param_groups[0]["lr"]
    assert resumed.model.a.weight is resumed.model.b.weight


@pytest.mark.parametrize("device", _DEVICES)
def test_a_restore_without_the_checkpoint_would_differ(device: str) -> None:
    # The resume test above must be able to fail: skipping the optimizer's
    # state changes the continuation.
    live = _Run(device, lucid.float32, "Adam", False)
    for k in range(3):
        live.step(k)
    ck = live.checkpoint()
    assert isinstance(ck, dict)
    tail = [live.step(k) for k in range(3, 6)]

    forgetful = _Run(device, lucid.float32, "Adam", False)
    forgetful.model.load_state_dict(ck["model"])  # type: ignore[arg-type]
    assert [forgetful.step(k) for k in range(3, 6)] != tail


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("opt", ["SGD", "Adam"])
def test_optimizer_state_is_cast_to_the_parameters_device(device: str, opt: str) -> None:
    live = _Run(device, lucid.float32, opt, False)
    for k in range(2):
        live.step(k)
    ck = _pickle_roundtrip(live.opt.state_dict())
    assert isinstance(ck, dict)
    other = _Run(device, lucid.float32, opt, False)
    other.opt.load_state_dict(ck)  # type: ignore[arg-type]
    for st in other.opt.state_dict()["state"].values():
        for v in st.values():
            if isinstance(v, lucid.Tensor):
                assert v.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize("name", _OPTS)
def test_the_optimizer_state_dict_has_the_references_keys(ref: ModuleType, name: str) -> None:
    kwargs = {
        "SGD": dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-3),
        "Adam": dict(lr=0.01, amsgrad=True),
        "AdamW": dict(lr=0.01),
        "RMSprop": dict(lr=0.01, momentum=0.5, centered=True),
        "Adagrad": dict(lr=0.05),
        "Adadelta": {},
        "Adamax": dict(lr=0.01),
        "NAdam": dict(lr=0.01),
        "RAdam": dict(lr=0.01),
        "ASGD": dict(lr=0.01),
    }[name]
    lm, rm = nn.Linear(3, 2), ref.nn.Linear(3, 2)
    lo = getattr(optim, name)(lm.parameters(), **kwargs)
    ro = getattr(ref.optim, name)(rm.parameters(), **kwargs)
    for _ in range(2):
        lo.zero_grad()
        ro.zero_grad()
        lm(lucid.randn(4, 3)).sum().backward()
        rm(ref.randn(4, 3)).sum().backward()
        lo.step()
        ro.step()
    ls, rs = lo.state_dict(), ro.state_dict()
    assert set(ls["state"][0]) == set(rs["state"][0])
    assert set(rs["param_groups"][0]) <= set(ls["param_groups"][0])
    assert ls["param_groups"][0]["params"] == rs["param_groups"][0]["params"]


def test_a_checkpoint_is_plain_data_under_weights_only() -> None:
    # A training checkpoint — model, optimizer, scheduler, scaler — loads with
    # the default (safe) unpickler; nothing in it needs arbitrary code.
    run = _Run("cpu", lucid.float32, "AdamW", True)
    for k in range(2):
        run.step(k)
    buf = io.BytesIO()
    save(
        {
            "model": run.model.state_dict(),
            "opt": run.opt.state_dict(),
            "sched": run.sched.state_dict() if run.sched else None,
            "scaler": run.scaler.state_dict(),
        },
        buf,
    )
    buf.seek(0)
    assert isinstance(load(buf, weights_only=True), dict)
    buf.seek(0)
    with pytest.raises(pickle.UnpicklingError):
        load(io.BytesIO(pickle.dumps(_Tied())), weights_only=True)
