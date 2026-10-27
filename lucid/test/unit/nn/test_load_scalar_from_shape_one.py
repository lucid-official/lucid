"""A 0-d parameter or buffer loads a shape ``(1,)`` checkpoint entry as its scalar (LCD-504).

Checkpoints written before 0-d tensors existed stored a scalar as shape
``(1,)``, and the reference still loads them: a 0-d slot takes the single
element, on a copy load and on an assign load alike.  Lucid refused it as a
size mismatch.  Only shape ``(1,)`` is taken — ``(3,)``, ``(1, 1)`` and
``(0,)`` stay mismatches, there as here.
"""

import pytest

import lucid
import lucid.nn as nn


def _module(lib, device=None):
    m = lib.nn.Module()
    m.p = lib.nn.Parameter(lib.tensor(1.0, device=device))
    m.register_buffer("b", lib.tensor(2, device=device))
    return m


@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
def test_a_scalar_slot_takes_a_one_element_entry(device, assign):
    m = _module(lucid, device)
    p = m.p
    m.load_state_dict({"p": lucid.tensor([5.0]), "b": lucid.tensor([9])}, assign=assign)
    assert m.p.shape == () and m.p.item() == 5.0
    assert m.b.shape == () and m.b.item() == 9 and m.b.dtype == lucid.int64
    assert type(m.p) is nn.Parameter and m.p.requires_grad
    assert (m.p is p) is not assign  # a copy writes into the old object


@pytest.mark.parametrize("shape", [(3,), (1, 1), (0,)], ids=["three", "one-by-one", "empty"])
@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
def test_any_other_shape_is_still_a_mismatch(shape, assign):
    m = _module(lucid)
    with pytest.raises(RuntimeError, match="size mismatch for p"):
        m.load_state_dict({"p": lucid.zeros(*shape), "b": lucid.tensor(9)}, assign=assign)
    assert m.p.item() == 1.0  # refused, so left as it was


def test_a_one_element_slot_is_not_squeezed_the_other_way():
    m = nn.Module()
    m.p = nn.Parameter(lucid.tensor([1.0]))
    with pytest.raises(RuntimeError, match="size mismatch"):
        m.load_state_dict({"p": lucid.tensor(5.0)})


@pytest.mark.parity
@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
@pytest.mark.parametrize(
    "shape,loads",
    [((1,), True), ((3,), False), ((1, 1), False), ((0,), False)],
    ids=["one", "three", "one-by-one", "empty"],
)
def test_the_reference_loads_the_same_entries(ref, assign, shape, loads):
    for lib, m in ((lucid, _module(lucid)), (ref, _module(ref))):
        state = {"p": lib.full(shape, 5.0), "b": lib.tensor(9)}
        if loads:
            m.load_state_dict(state, assign=assign)
            assert tuple(m.p.shape) == () and m.p.item() == 5.0, lib.__name__
        else:
            with pytest.raises(RuntimeError, match="size mismatch"):
                m.load_state_dict(state, assign=assign)
