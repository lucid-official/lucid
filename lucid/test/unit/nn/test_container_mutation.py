"""Deleting from and inserting into the module containers.

``nn/modules/container.py`` sat at 52.4%, and the dark part was every
mutation: ``__delitem__`` on all three containers and ``insert`` on
``Sequential`` and ``ModuleList``.  Construction was covered; changing
one afterwards was not.

The containers key their children by *position stringified* — ``"0"``,
``"1"``, ``"2"``.  So a deletion has to renumber everything after it,
and an insertion has to shift everything from the index up.  Getting
that wrong leaves a container whose ``len`` is right, whose ``forward``
runs, and whose ``state_dict`` keys no longer name the layers they used
to — which turns every checkpoint written before the edit into one that
loads into the wrong layers.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _v(x):
    return np.asarray(x.numpy())


def _names(container):
    return [type(m).__name__ for m in container]


def _keys(container):
    return list(dict(container.named_children()))


def _x(shape=(2, 4)):
    return lucid.tensor(np.ones(shape, dtype=np.float32))


# ── deleting ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [lambda mods: nn.Sequential(*mods), lambda mods: nn.ModuleList(mods)],
    ids=["sequential", "modulelist"],
)
def test_deleting_removes_the_module_and_renumbers_the_rest(build):
    container = build([nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid()])
    del container[1]
    assert _names(container) == ["Linear", "Sigmoid"]
    assert len(container) == 2
    assert _keys(container) == ["0", "1"], "a gap in the keys breaks every checkpoint"


@pytest.mark.parametrize(
    "build",
    [lambda mods: nn.Sequential(*mods), lambda mods: nn.ModuleList(mods)],
    ids=["sequential", "modulelist"],
)
def test_deleting_a_slice(build):
    container = build([nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid(), nn.Tanh()])
    del container[1:3]
    assert _names(container) == ["Linear", "Tanh"]
    assert _keys(container) == ["0", "1"]


def test_deleting_the_first_and_the_last():
    container = nn.ModuleList([nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid()])
    del container[0]
    assert _names(container) == ["ReLU", "Sigmoid"]
    del container[-1]
    assert _names(container) == ["ReLU"]
    assert _keys(container) == ["0"]


def test_a_sequential_still_runs_after_a_deletion():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    del model[1]
    assert _v(model(_x())).shape == (2, 2)


def test_the_parameters_go_with_the_deleted_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    before = len(list(model.parameters()))
    del model[0]
    assert len(list(model.parameters())) == before - 2


def test_the_state_dict_keys_follow_the_renumbering():
    """The failure that outlives the process: keys that no longer name
    the layers a saved checkpoint was written against."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    del model[1]
    assert sorted(model.state_dict()) == ["0.bias", "0.weight", "1.bias", "1.weight"]


def test_deleting_past_the_end_is_refused():
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises((IndexError, KeyError)):
        del model[5]


def test_deleting_from_a_module_dict_removes_the_key():
    container = nn.ModuleDict({"encoder": nn.Linear(4, 4), "head": nn.ReLU()})
    del container["encoder"]
    assert list(container) == ["head"]
    assert "encoder" not in container


def test_deleting_an_absent_key_from_a_module_dict_is_refused():
    container = nn.ModuleDict({"head": nn.ReLU()})
    with pytest.raises(KeyError):
        del container["missing"]


# ── a compiled child survives delete + renumber (LCD-571) ─────────────────────


@pytest.mark.parametrize(
    "build",
    [lambda mods: nn.Sequential(*mods), lambda mods: nn.ModuleList(mods)],
    ids=["sequential", "modulelist"],
)
def test_deleting_before_a_compiled_child_keeps_it_compiled(build):
    """``_renumber_modules`` used to clear ``_modules`` and re-add through
    ``add_module``, leaving the old key's ``_ModuleWrapper`` behind in
    ``__dict__`` -- clashing with whatever key the shift assigned next and
    raising ``KeyError`` partway through, with the remaining children lost."""
    container = build([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2)])
    container[1] = lucid.compile(container[1])
    del container[0]
    assert _keys(container) == ["0", "1"]
    assert isinstance(container.__dict__.get("0"), lucid.nn.module._ModuleWrapper)
    assert "1" not in container.__dict__, "stale wrapper left under the old key"
    out = container[0](_x())
    for m in list(container)[1:]:
        out = m(out)
    assert _v(out).shape == (2, 2)


def test_deleting_from_a_module_dict_clears_a_compiled_childs_wrapper():
    container = nn.ModuleDict({"a": nn.Linear(4, 4), "b": nn.Linear(4, 4)})
    container.b = lucid.compile(container.b)
    del container["b"]
    assert "b" not in container.__dict__
    assert "b" not in dict(container.named_children())


# ── inserting ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [lambda mods: nn.Sequential(*mods), lambda mods: nn.ModuleList(mods)],
    ids=["sequential", "modulelist"],
)
def test_inserting_shifts_everything_after_it(build):
    container = build([nn.Linear(4, 4), nn.Sigmoid()])
    container.insert(1, nn.ReLU())
    assert _names(container) == ["Linear", "ReLU", "Sigmoid"]
    assert _keys(container) == ["0", "1", "2"]


def test_inserting_at_the_front_and_at_the_end():
    container = nn.ModuleList([nn.ReLU()])
    container.insert(0, nn.Linear(4, 4))
    container.insert(2, nn.Sigmoid())
    assert _names(container) == ["Linear", "ReLU", "Sigmoid"]
    assert _keys(container) == ["0", "1", "2"]


def test_inserting_at_a_negative_index():
    container = nn.ModuleList([nn.Linear(4, 4), nn.Sigmoid()])
    container.insert(-1, nn.ReLU())
    assert _names(container) == ["Linear", "ReLU", "Sigmoid"]


def test_inserting_registers_the_new_parameters():
    container = nn.ModuleList([nn.ReLU()])
    before = len(list(container.parameters()))
    container.insert(0, nn.Linear(4, 4))
    assert len(list(container.parameters())) == before + 2


def test_a_sequential_runs_the_inserted_module():
    """Inserted in the middle, so a container that appended instead would
    still have the right ``len`` and the wrong answer."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    without = _v(model(_x()))
    model.insert(1, nn.ReLU())
    assert _names(model) == ["Linear", "ReLU", "Linear"]
    assert not np.allclose(_v(model(_x())), without)


def test_inserting_then_deleting_returns_the_original():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    before = _v(model(_x()))
    model.insert(1, nn.Identity())
    del model[1]
    assert _names(model) == ["Linear", "Linear"]
    assert _keys(model) == ["0", "1"]
    assert np.allclose(_v(model(_x())), before)


# ── the containers still behave as containers ─────────────────────────────────


def test_a_mutated_container_still_trains():
    model = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid(), nn.Linear(4, 2))
    del model[1]
    model.insert(1, nn.ReLU())
    optimiser = lucid.optim.SGD(model.parameters(), lr=0.1)
    first = float(_v((model(_x()) ** 2).mean()))
    for _ in range(10):
        optimiser.zero_grad()
        (model(_x()) ** 2).mean().backward()
        optimiser.step()
    assert float(_v((model(_x()) ** 2).mean())) < first


def test_a_mutated_container_round_trips_through_a_checkpoint():
    source = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    del source[1]

    target = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    target.load_state_dict(source.state_dict())
    assert np.allclose(_v(source(_x())), _v(target(_x())), atol=1e-6)


def test_iteration_and_indexing_agree_after_a_mutation():
    container = nn.ModuleList([nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid()])
    del container[1]
    container.insert(0, nn.Tanh())
    assert [type(m).__name__ for m in container] == [
        type(container[i]).__name__ for i in range(len(container))
    ]
