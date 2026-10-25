"""``ParameterList`` / ``ParameterDict`` entry points that skipped the wrap,
and an assert that crashed on a slot the reference reads back as ``None``
(LCD-571).

Only ``ParameterList.__setitem__`` wrapped a plain tensor into a
:class:`~lucid.nn.Parameter`; ``append`` / ``extend`` / ``__init__`` and
every :class:`~lucid.nn.ParameterDict` entry point (``__setitem__`` /
``__init__`` / ``update``) registered the bare tensor and let
``register_parameter`` refuse it with a ``TypeError`` instead.  Separately,
``__getitem__`` (and, for ``ParameterList``, ``__iter__``) asserted the
stored value was not ``None`` -- which a slot explicitly cleared by
assignment legitimately is, in both frameworks.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn

# ── plain tensors get wrapped, at every entry point ─────────────────────────


def test_parameter_list_wraps_a_plain_tensor_on_append():
    params = nn.ParameterList()
    params.append(lucid.zeros(2))
    assert type(params[0]) is nn.Parameter


def test_parameter_list_wraps_plain_tensors_on_extend():
    params = nn.ParameterList()
    params.extend([lucid.zeros(2), nn.Parameter(lucid.ones(2))])
    assert type(params[0]) is nn.Parameter
    assert type(params[1]) is nn.Parameter


def test_parameter_list_wraps_a_plain_tensor_at_construction():
    params = nn.ParameterList([lucid.zeros(2)])
    assert type(params[0]) is nn.Parameter


def test_parameter_dict_wraps_a_plain_tensor_on_setitem():
    params = nn.ParameterDict()
    params["a"] = lucid.zeros(2)
    assert type(params["a"]) is nn.Parameter


def test_parameter_dict_wraps_a_plain_tensor_at_construction():
    params = nn.ParameterDict({"a": lucid.zeros(2)})
    assert type(params["a"]) is nn.Parameter


def test_parameter_dict_wraps_plain_tensors_on_update():
    params = nn.ParameterDict()
    params.update({"a": lucid.zeros(2)})
    assert type(params["a"]) is nn.Parameter


def test_an_appended_tensor_is_trained_like_any_parameter():
    params = nn.ParameterList()
    params.append(lucid.ones(2))
    assert params[0] is not None
    optimiser = lucid.optim.SGD(params.parameters(), lr=0.1)
    (params[0] ** 2).sum().backward()
    before = float(params[0][0])
    optimiser.step()
    assert float(params[0][0]) < before


# ── a non-tensor, non-Parameter value is still refused ──────────────────────


def test_parameter_list_append_still_refuses_a_non_tensor():
    with pytest.raises(TypeError):
        nn.ParameterList().append("not a tensor")


def test_parameter_dict_setitem_still_refuses_a_non_tensor():
    with pytest.raises(TypeError):
        nn.ParameterDict()["a"] = "not a tensor"


# ── a slot explicitly cleared to None reads back as None, not an assert ─────


def test_parameter_list_getitem_returns_none_for_a_cleared_slot():
    params = nn.ParameterList([nn.Parameter(lucid.zeros(2))])
    params[0] = None
    assert params[0] is None


def test_parameter_list_iteration_includes_a_cleared_slot():
    params = nn.ParameterList(
        [nn.Parameter(lucid.zeros(2)), nn.Parameter(lucid.ones(2))]
    )
    params[0] = None
    seen = list(params)
    assert seen[0] is None
    assert seen[1] is not None


def test_parameter_dict_getitem_returns_none_for_a_cleared_slot():
    params = nn.ParameterDict({"a": nn.Parameter(lucid.zeros(2))})
    params["a"] = None
    assert params["a"] is None


def test_parameter_dict_pop_returns_none_for_a_cleared_slot():
    params = nn.ParameterDict({"a": nn.Parameter(lucid.zeros(2))})
    params["a"] = None
    assert params.pop("a") is None
    assert "a" not in params


# ── parity: the reference wraps and reads back None the same way ───────────


@pytest.mark.parity
def test_wrapping_and_none_readback_match_the_reference(ref: ModuleType):
    got = nn.ParameterList()
    got.append(lucid.zeros(2))
    got[0] = None

    want = ref.nn.ParameterList()
    want.append(ref.zeros(2))
    want[0] = None

    assert type(got[0]) is type(None) is type(want[0])

    got2 = nn.ParameterList([lucid.ones(2)])
    want2 = ref.nn.ParameterList([ref.ones(2)])
    assert type(got2[0]).__name__ == type(want2[0]).__name__ == "Parameter"
