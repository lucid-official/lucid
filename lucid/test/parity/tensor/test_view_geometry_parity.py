"""Parity guards for view storage geometry against the reference (LCD-517 Q1).

Checks the ``(view op x position)`` grid from
:mod:`lucid.test.unit.tensor.test_view_geometry_contract` against the reference
framework (via the ``ref`` fixture, with no NumPy dependency):

* Values, ``storage_offset()``, moving strides (axes of length > 1), and
  ``is_contiguous()`` on CPU and Metal.
* ``untyped_storage().nbytes()`` and the ``data_ptr()`` offset relation
  ``v.data_ptr() - v.untyped_storage().data_ptr() == v.storage_offset() *
  itemsize``.
* ``as_strided`` windows against the root at offsets ``{0, own, last valid}``
  and a ``fill_`` write through the window.
"""

from typing import Any

import pytest

import lucid
from lucid.test.unit.tensor.test_view_geometry_contract import (
    GRID,
    _DEVICES,
    _moving,
)

pytestmark = pytest.mark.parity


def _apply_named(name: str, root: Any, module: Any) -> Any:
    """Apply the view operation ``name`` to a Lucid or reference tensor."""
    if name == "conj-first":
        return module.conj(root[0])
    if name == "conj-middle":
        return module.conj(root[1])
    if name == "conj-last":
        return module.conj(root[-1])
    return GRID[name].make(root)


def _layout_parity_params() -> list[object]:
    params: list[object] = []
    for name in GRID:
        for device in _DEVICES:
            params.append(pytest.param(name, device, id=f"{name}-{device}"))
    return params


@pytest.mark.parametrize(("name", "device"), _layout_parity_params())
def test_view_layout_matches_the_reference(name: str, device: str, ref: Any) -> None:
    ours_root = lucid.arange(24.0, device=device).reshape(4, 6)
    theirs_root = ref.arange(24.0).reshape(4, 6)
    ours = _apply_named(name, ours_root, lucid)
    theirs = _apply_named(name, theirs_root, ref)
    assert tuple(ours.shape) == tuple(theirs.shape)
    assert ours.tolist() == theirs.tolist()
    assert _moving(tuple(ours.shape), tuple(ours.stride())) == _moving(
        tuple(theirs.shape), tuple(theirs.stride())
    )
    assert ours.is_contiguous() == theirs.is_contiguous()
    assert ours.storage_offset() == theirs.storage_offset()


@pytest.mark.parametrize("name", list(GRID))
def test_untyped_storage_and_data_ptr_match_the_reference_on_cpu(name: str, ref: Any) -> None:
    ours_root = lucid.arange(24.0).reshape(4, 6)
    theirs_root = ref.arange(24.0).reshape(4, 6)
    ours = _apply_named(name, ours_root, lucid)
    theirs = _apply_named(name, theirs_root, ref)
    assert ours.untyped_storage().nbytes() == theirs.untyped_storage().nbytes()
    assert ours.untyped_storage().data_ptr() == ours_root.untyped_storage().data_ptr()
    ours_delta = ours.data_ptr() - ours.untyped_storage().data_ptr()
    assert ours_delta == ours.storage_offset() * ours.element_size()
    assert ours_delta == theirs.storage_offset() * theirs.element_size()
    if ours.numel() > 0:
        theirs_delta = theirs.data_ptr() - theirs.untyped_storage().data_ptr()
        assert ours_delta == theirs_delta


def _window_parity_params() -> list[object]:
    params: list[object] = []
    for name in GRID:
        for offset_kind in ("zero", "own", "last"):
            for device in _DEVICES:
                params.append(
                    pytest.param(
                        name,
                        offset_kind,
                        device,
                        id=f"{name}-{offset_kind}-{device}",
                    )
                )
    return params


def _pick_offset(offset_kind: str, view_offset: int, root_numel: int) -> int:
    if offset_kind == "zero":
        return 0
    if offset_kind == "own":
        return min(view_offset, root_numel - 2)
    return root_numel - 2


@pytest.mark.parametrize(("name", "offset_kind", "device"), _window_parity_params())
def test_as_strided_windows_and_fill_match_the_reference(
    name: str, offset_kind: str, device: str, ref: Any
) -> None:
    ours_root = lucid.arange(24.0, device=device).reshape(4, 6)
    theirs_root = ref.arange(24.0).reshape(4, 6)
    theirs_view = _apply_named(name, theirs_root, ref)
    offset = _pick_offset(offset_kind, int(theirs_view.storage_offset()), 24)
    want_win = theirs_view.as_strided((2,), (1,), offset)

    ours_view = _apply_named(name, ours_root, lucid)
    got_win = ours_view.as_strided((2,), (1,), offset)
    assert got_win.tolist() == want_win.tolist()

    # Drop the intermediate view handle before fill_ so a window write to the
    # root storage does not trip LCD-535's live-sibling refusal on Metal.
    del ours_view, theirs_view
    got_win.fill_(-9.0)
    want_win.fill_(-9.0)
    assert ours_root.tolist() == theirs_root.tolist()
