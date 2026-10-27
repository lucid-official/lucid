"""``lucid.save`` keeps every view's values and the storage views share (LCD-460).

When this was found, most Metal views were copies that a Python layer linked
back to their base, and the base a link named could be gone; a Metal view is
now a member of its base's view family in the engine, as a CPU view is.  The
saver used to lay a materialised view's packed bytes out as if they had its
base's layout, so a transpose, a column slice or a view of a view came back
scrambled — and scrambled its base too when both were saved.

Every case here is checked bit for bit against the tensor that was saved,
and, where the views were saved with what they view, by writing through one
loaded tensor and reading the other: views saved together share a storage
after a load, as they do in the reference.
"""

import gc
import io
import pickle
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import device_dtype_params

#: Each view of a (4, 6) base, and whether it reads an element twice.
_VIEWS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], bool]] = {
    "transpose": (lambda x: x.t(), False),
    "column": (lambda x: x[:, 2], False),
    "step_rows": (lambda x: x[::2], False),
    "step_columns": (lambda x: x[:, 1::2], False),
    "expand": (lambda x: x[None].expand(2, 4, 6), True),
    "narrow": (lambda x: x.narrow(1, 1, 3), False),
    "view_of_transpose": (lambda x: x.t()[1:4], False),
    "reshape_of_rows": (lambda x: x[1:3].reshape(-1), False),
    "step_of_flat": (lambda x: x.view(-1)[2:20:3], False),
}


def _base(device: str) -> lucid.Tensor:
    # Distinct bit patterns, including the ones an arithmetic round trip
    # would launder: -0.0, a NaN, the infinities and a subnormal.
    values = [float(i) * 1.25 - 9.0 for i in range(24)]
    values[3], values[8], values[13], values[17], values[22] = (
        -0.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        1e-45,
    )
    return lucid.tensor(values, device=device).reshape(4, 6)


def _bits(t: lucid.Tensor) -> bytes:
    return np.ascontiguousarray(t.numpy()).tobytes()


def _roundtrip(obj: object) -> object:
    buf = io.BytesIO()
    lucid.save(obj, buf)
    buf.seek(0)
    return lucid.load(buf)


def _marker(like: lucid.Tensor, start: float) -> lucid.Tensor:
    flat = lucid.arange(like.numel(), dtype=like.dtype, device=like.device)
    return (flat * -3.0 + start).reshape(like.shape)


def _assert_shares(
    base: lucid.Tensor,
    view: lucid.Tensor,
    make: Callable[[lucid.Tensor], lucid.Tensor],
    overlapping: bool,
) -> None:
    """A write through either loaded tensor is read through the other."""
    marker = _marker(base, 0.5)
    with lucid.no_grad():
        base.copy_(marker)
    assert _bits(view) == _bits(make(marker))
    if overlapping:
        return  # a write into an expanded view is refused, as on the CPU
    oracle = marker.to("cpu").clone()
    patch = _marker(view, 1000.0)
    with lucid.no_grad():
        make(oracle).copy_(patch.to("cpu"))
        view.copy_(patch)
    assert _bits(base) == _bits(oracle)


@pytest.mark.parametrize("kind", list(_VIEWS))
class TestViewRoundTrip:
    def test_alone_keeps_its_values(self, device: str, kind: str) -> None:
        make, _ = _VIEWS[kind]
        view = make(_base(device))
        r = _roundtrip({"v": view})
        assert isinstance(r, dict)
        assert r["v"].shape == view.shape
        assert r["v"].device == view.device
        assert _bits(r["v"]) == _bits(view)

    def test_with_its_base_shares_its_storage(self, device: str, kind: str) -> None:
        make, overlapping = _VIEWS[kind]
        base = _base(device)
        view = make(base)
        want_base, want_view = _bits(base), _bits(view)
        r = _roundtrip({"base": base, "v": view})
        assert isinstance(r, dict)
        assert _bits(r["base"]) == want_base
        assert _bits(r["v"]) == want_view
        _assert_shares(r["base"], r["v"], make, overlapping)

    def test_in_nested_containers_shares_its_storage(self, device: str, kind: str) -> None:
        make, overlapping = _VIEWS[kind]
        base = _base(device)
        view = make(base)
        want_base, want_view = _bits(base), _bits(view)
        r = _roundtrip({"outer": [base, {"inner": (view,)}]})
        assert isinstance(r, dict)
        loaded_base = r["outer"][0]
        loaded_view = r["outer"][1]["inner"][0]
        assert _bits(loaded_base) == want_base
        assert _bits(loaded_view) == want_view
        _assert_shares(loaded_base, loaded_view, make, overlapping)

    def test_taken_before_its_base_was_written(self, device: str, kind: str) -> None:
        # On Metal the view does not see the later write and holds the old
        # values; whatever it holds is what must come back.
        make, _ = _VIEWS[kind]
        base = _base(device)
        view = make(base)
        with lucid.no_grad():
            base.copy_(_marker(base, 7.0))
        want_base, want_view = _bits(base), _bits(view)
        r = _roundtrip({"base": base, "v": view})
        assert isinstance(r, dict)
        assert _bits(r["base"]) == want_base
        assert _bits(r["v"]) == want_view


def test_views_whose_base_is_gone_keep_their_values(device: str) -> None:
    base = lucid.tensor([float(i) for i in range(24)], device=device).reshape(4, 6)
    flipped, row = base.t(), base[1]
    del base
    gc.collect()
    want = _bits(flipped), _bits(row)
    r = _roundtrip({"t": flipped, "row": row})
    assert isinstance(r, dict)
    assert (_bits(r["t"]), _bits(r["row"])) == want


def test_overlapping_slices_without_their_base_share(device: str) -> None:
    base = _base(device).reshape(-1)
    r = _roundtrip({"a": base[2:12], "b": base[6:20]})
    assert isinstance(r, dict)
    assert _bits(r["a"]) == _bits(base[2:12])
    assert _bits(r["b"]) == _bits(base[6:20])
    with lucid.no_grad():
        r["a"].copy_(_marker(r["a"], 3.0))
    assert _bits(r["b"][:6]) == _bits(_marker(r["a"], 3.0)[4:])


def test_a_clone_does_not_share_with_its_source(device: str) -> None:
    # On Metal a clone reads the same MLX buffer as its source until either
    # is written; it is still a tensor of its own.
    x = _base(device)
    c = x.clone()
    r = _roundtrip({"x": x, "c": c})
    assert isinstance(r, dict)
    with lucid.no_grad():
        r["x"].copy_(_marker(x, 0.0))
    assert _bits(r["c"]) == _bits(c)


def test_a_view_is_saved_once_with_its_base(device: str) -> None:
    base = lucid.randn(64, 64, device=device)
    alone, both = io.BytesIO(), io.BytesIO()
    lucid.save({"base": base}, alone)
    lucid.save({"base": base, "t": base.t(), "col": base[:, 3]}, both)
    assert len(both.getvalue()) < len(alone.getvalue()) + 512


_ALL_DTYPES = (
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool_,
    lucid.complex64,
    lucid.complex128,
)


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_ALL_DTYPES))
def test_every_dtype_shares_and_keeps_its_values(device: str, dtype: lucid.dtype) -> None:
    counts = lucid.arange(24, dtype=lucid.int64).reshape(4, 6)
    source = counts % 3 == 0 if dtype == lucid.bool_ else counts.to(dtype)
    base = source.to(device)
    flat = base.reshape(-1)
    # A base with its view, overlapping views with no base, and a view alone:
    # the storage is copied whole, assembled from its views, or not shared.
    cases = [{"base": base, "col": base[:, 2]}, {"a": flat[2:12], "b": flat[6:20]}, {"t": base.t()}]
    for case in cases:
        r = _roundtrip(case)
        assert isinstance(r, dict)
        for key, t in case.items():
            assert r[key].dtype == dtype
            assert r[key].tolist() == t.tolist(), key


def test_weights_copied_into_another_model_stay_apart(device: str) -> None:
    # On Metal ``load_state_dict`` leaves the copy reading the source's MLX
    # buffer; the two models are still two models after a load.
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    ema = nn.Linear(4, 3).to(device)
    ema.load_state_dict(model.state_dict())
    r = _roundtrip({"model": model.state_dict(), "ema": ema.state_dict()})
    assert isinstance(r, dict)
    before = _bits(r["ema"]["weight"])
    assert before == _bits(model.weight)
    with lucid.no_grad():
        r["model"]["weight"].add_(1.0)
    assert _bits(r["ema"]["weight"]) == before


class TestParameters:
    def test_parameter_of_a_tensor_keeps_its_values(self, device: str) -> None:
        x = _base(device)
        p = nn.Parameter(x)
        r = _roundtrip({"x": x, "p": p})
        assert isinstance(r, dict)
        assert type(r["p"]) is nn.Parameter
        assert r["p"].requires_grad is True
        assert type(r["x"]) is lucid.Tensor
        assert _bits(r["p"]) == _bits(x)
        assert _bits(r["x"]) == _bits(x)

    def test_parameter_of_a_tensor_shares_with_it(self, device: str) -> None:
        x = _base(device)
        r = _roundtrip({"x": x, "p": nn.Parameter(x)})
        assert isinstance(r, dict)
        with lucid.no_grad():
            r["x"].copy_(_marker(x, 2.0))
        assert _bits(r["p"]) == _bits(_marker(x, 2.0))

    def test_engine_family_members_survive_repeated_roundtrip(self, device: str) -> None:
        x = _base(device)
        r1 = _roundtrip({"x": x, "p": nn.Parameter(x), "f": nn.Flatten(0)(x)})
        r = _roundtrip(r1)
        assert isinstance(r, dict)
        marker = _marker(x, 4.0)
        with lucid.no_grad():
            r["x"].copy_(marker)
        assert _bits(r["p"]) == _bits(marker)
        assert _bits(r["f"]) == _bits(marker.reshape(-1))

    def test_parameter_of_a_view_alone_trains_after_load(self, device: str) -> None:
        # A view saved alone loads dense, owning its elements.  (Stepping a
        # strided Parameter is its own defect, LCD-476.)
        x = lucid.arange(24.0, device=device).reshape(4, 6)
        p = nn.Parameter(x[:, 2])
        r = _roundtrip({"p": p})
        assert isinstance(r, dict)
        q = r["p"]
        assert type(q) is nn.Parameter and q.requires_grad is True
        assert q.tolist() == [2.0, 8.0, 14.0, 20.0]
        opt = lucid.optim.SGD([q], lr=0.5)
        (q * 2.0).sum().backward()
        opt.step()
        assert q.tolist() == [1.0, 7.0, 13.0, 19.0]

    def test_tied_views_of_a_parameter(self, device: str) -> None:
        w = nn.Parameter(_base(device))
        r = _roundtrip({"w": w, "wt": w.t(), "again": w})
        assert isinstance(r, dict)
        assert r["again"] is r["w"]
        assert type(r["w"]) is nn.Parameter
        assert _bits(r["wt"]) == _bits(w.t())


class _Pid:
    """Stands in for a tensor while a legacy file is written."""

    def __init__(self, pid: tuple[object, ...]) -> None:
        self.pid = pid


class _LegacyPickler(pickle.Pickler):
    def persistent_id(self, obj: object) -> object:
        return obj.pid if isinstance(obj, _Pid) else None


def _legacy_file(container: dict[str, object]) -> io.BytesIO:
    buf = io.BytesIO()
    _LegacyPickler(buf, protocol=4).dump(container)
    buf.seek(0)
    return buf


_SIX = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
_SIX_BYTES = np.asarray(_SIX, dtype=np.float32).tobytes()


class TestOlderFiles:
    def test_format_1_numpy_dtype_records(self) -> None:
        pid = ("tensor", [2, 3], "float32", "cpu", _SIX_BYTES, "float32")
        r = lucid.load(_legacy_file({"_lucid_format": 1, "obj": {"w": _Pid(pid)}}))
        assert isinstance(r, dict)
        assert r["w"].tolist() == [_SIX[:3], _SIX[3:]]

    def test_format_2_dense_records(self) -> None:
        pid = ("tensor_v3", [3, 2], "float32", "cpu", _SIX_BYTES)
        r = lucid.load(_legacy_file({"_lucid_format": 2, "obj": [_Pid(pid)]}))
        assert isinstance(r, list)
        assert r[0].tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]

    @pytest.mark.parametrize("target", ["cpu", "metal"])
    def test_format_2_shared_storage_records(self, target: str) -> None:
        if target == "metal" and not lucid.metal.is_available():
            pytest.skip("no Metal device")
        base = ("tensor_v3", 0, [2, 3], [3, 1], 0, "float32", "cpu", False, False)
        flipped = ("tensor_v3", 1, [3, 2], [1, 3], 0, "float32", "cpu", True, True)
        r = lucid.load(
            _legacy_file(
                {
                    "_lucid_format": 2,
                    "obj": {"base": _Pid((*base, 0, _SIX_BYTES)), "t": _Pid((*flipped, 0, None))},
                }
            ),
            map_location=target,
        )
        assert isinstance(r, dict)
        assert r["t"].tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
        assert type(r["t"]) is nn.Parameter and r["t"].requires_grad is True
        with lucid.no_grad():
            r["base"][0, 2] = 55.0
        assert r["t"][2, 0].item() == 55.0

    def test_a_record_without_its_storage_is_refused(self) -> None:
        pid = ("tensor_v3", 0, [2], [1], 0, "float32", "cpu", False, False, 4, None)
        damaged = _legacy_file({"_lucid_format": 2, "obj": [_Pid(pid)]})
        with pytest.raises(pickle.UnpicklingError, match="storage 4 has no bytes"):
            lucid.load(damaged)
