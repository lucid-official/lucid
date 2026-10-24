"""Layout probes must preserve dtype and observe both complex components."""

import numpy as np
import pytest

import lucid
from lucid.test.audit import _specs, _surface
from lucid.test.audit._axes import Context
from lucid.test.audit._axes_stability import ContiguityAxis
from lucid.test.audit._result import Status
from lucid.test._fixtures.devices import metal_available


@pytest.mark.parametrize(
    "dtype", [lucid.int64, lucid.float16, lucid.complex64, lucid.complex128]
)
def test_layout_probe_preserves_values_dtype_and_uses_an_offset_view(
    monkeypatch: pytest.MonkeyPatch,
    dtype: lucid.dtype,
) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]])
    if dtype in (lucid.complex64, lucid.complex128):
        values = values + 1j * values
    original = lucid.tensor(values, dtype=dtype)
    call = _specs.Call([original])
    seen_views = []

    def identity(x: lucid.Tensor) -> lucid.Tensor:
        assert x.dtype == dtype
        np.testing.assert_array_equal(x.numpy(), values)
        if not x.is_contiguous():
            seen_views.append(x.storage_offset())
        return x.clone()

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    finding = axis.run(
        _surface.Symbol("lucid.identity", "lucid", "op", identity), Context()
    )
    assert finding.status == Status.PASS, finding.detail
    assert seen_views and all(offset > 0 for offset in seen_views)


def test_layout_probe_detects_an_imaginary_only_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _specs.Call([lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.complex64)])

    def corrupt_imaginary(x: lucid.Tensor) -> lucid.Tensor:
        out = x.clone()
        if not x.is_contiguous():
            out = out + lucid.tensor([1j, 1j, 1j], dtype=lucid.complex64)
        return out

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    symbol = _surface.Symbol("lucid.corrupt", "lucid", "op", corrupt_imaginary)
    finding = axis.run(symbol, Context())
    assert finding.status == Status.FAIL, finding.detail


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_metal_layout_probe_hands_the_op_a_strided_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Metal slices are strided members of their base's view family
    # (730ef06c5), so the layout half has a strided operand there too and
    # its verdict is not the vacuous one.
    original = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="metal")
    call = _specs.Call([original])
    handed: list[bool] = []

    def copy(x: lucid.Tensor) -> lucid.Tensor:
        handed.append(bool(x._impl.is_contiguous()))
        return x.clone()

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    finding = axis.run(_surface.Symbol("lucid.copy", "lucid", "op", copy), Context())
    assert finding.status == Status.PASS, finding.detail
    assert finding.detail == "strided and packed agree"
    assert False in handed


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_metal_layout_probe_catches_an_op_that_misreads_a_strided_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="metal")
    call = _specs.Call([original])

    def misread(x: lucid.Tensor) -> lucid.Tensor:
        out = x.clone()
        if not x._impl.is_contiguous():
            out = out + 1.0
        return out

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    symbol = _surface.Symbol("lucid.misread", "lucid", "op", misread)
    finding = axis.run(symbol, Context())
    assert finding.status == Status.FAIL, finding.detail


def test_a_slice_that_comes_back_packed_makes_the_layout_half_vacuous(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    # No device copies a positive-step slice any more; the guard stays for
    # the day one does, so the axis never claims a layout check it did not
    # make.
    original = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    call = _specs.Call([original])
    getitem = lucid.Tensor.__getitem__
    monkeypatch.setattr(
        lucid.Tensor,
        "__getitem__",
        lambda self, key: getitem(self, key).contiguous(),
    )
    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    symbol = _surface.Symbol("lucid.copy", "lucid", "op", lambda x: x.clone())
    finding = axis.run(symbol, Context())
    assert finding.status == Status.PASS, finding.detail
    assert "vacuous" in finding.detail
