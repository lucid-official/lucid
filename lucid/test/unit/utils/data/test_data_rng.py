"""Randomness in ``lucid.utils.data`` comes from Lucid's generators — one way.

The samplers and ``random_split`` drew from Python's ``random``: some
from the module-level state, some from a ``random.Random`` rebuilt from
the same seed every epoch.  So :func:`lucid.manual_seed` reproduced
neither a shuffle nor a split, a :class:`lucid.Generator` raised
``TypeError``, and an ``int`` seed froze every epoch to the first one
(50 distinct indices over five epochs where a fresh draw gives ~90).

The guards here are class-wide rather than per symptom: every public
callable in ``lucid.utils.data`` that takes ``generator=`` is found by
introspection and held to the same contract, and no module of the
package may draw from Python's (or NumPy's) process-global RNG.
"""

import ast
import inspect
from pathlib import Path

import pytest

import lucid
import lucid.utils.data as data
from lucid.utils.data import (
    DataLoader,
    RandomSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    random_split,
)

from lucid.test.unit.utils.data._worker_datasets import Indices

_N = 24


@pytest.fixture(autouse=True)
def _on_default_device(device):
    """Every contract here holds whatever the default device is.

    With ``set_default_device("metal")`` the draws went to the GPU: every
    seed came back 0 (so each epoch repeated the last, and every worker
    pool got base seed 0), ``WeightedRandomSampler`` died on ``float64``,
    and the same seed shuffled differently on the two devices.  Sampling
    is host work and is now pinned there.
    """
    previous = lucid.get_default_device()
    lucid.set_default_device(device)
    try:
        yield device
    finally:
        lucid.set_default_device(previous)


def test_the_default_device_does_not_change_a_draw(device):
    """Same seed, same order — on whichever device the model lives."""
    draws = []
    for target in ("cpu", device):
        lucid.set_default_device(target)
        draws.append({name: EPOCHS[name](lucid.Generator(4))() for name in EPOCHS})
    assert draws[0] == draws[1]


def _loader_epochs(generator):
    loader = DataLoader(Indices(_N), batch_size=None, shuffle=True, generator=generator)
    return lambda: [int(i) for i in loader]


def _random_sampler_epochs(generator):
    sampler = RandomSampler(Indices(_N), generator=generator)
    return lambda: list(sampler)


def _random_sampler_replacement_epochs(generator):
    sampler = RandomSampler(Indices(_N), replacement=True, generator=generator)
    return lambda: list(sampler)


def _subset_epochs(generator):
    sampler = SubsetRandomSampler(list(range(100, 100 + _N)), generator=generator)
    return lambda: list(sampler)


def _weighted_epochs(generator):
    sampler = WeightedRandomSampler([1.0] * _N, _N, generator=generator)
    return lambda: list(sampler)


def _weighted_unique_epochs(generator):
    weights = [float(i + 1) for i in range(_N)]
    sampler = WeightedRandomSampler(weights, _N, replacement=False, generator=generator)
    return lambda: list(sampler)


def _split_epochs(generator):
    dataset = Indices(_N)
    return lambda: random_split(dataset, [_N // 2, _N // 2], generator=generator)[
        0
    ].indices


# Every public ``generator=`` taker, and how to draw one epoch from it.
EPOCHS = {
    "DataLoader": _loader_epochs,
    "RandomSampler": _random_sampler_epochs,
    "RandomSampler(replacement)": _random_sampler_replacement_epochs,
    "SubsetRandomSampler": _subset_epochs,
    "WeightedRandomSampler": _weighted_epochs,
    "WeightedRandomSampler(no replacement)": _weighted_unique_epochs,
    "random_split": _split_epochs,
}


def test_every_generator_taking_callable_is_held_to_the_contract():
    """A sampler added later with a ``generator=`` must join ``EPOCHS``."""
    takers = set()
    for name in data.__all__:
        obj = getattr(data, name)
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if "generator" in params:
            takers.add(name)
    covered = {key.split("(")[0] for key in EPOCHS}
    assert takers == covered


@pytest.mark.parametrize("name", sorted(EPOCHS))
def test_a_fixed_generator_reproduces_and_the_global_one_does_not_interfere(name):
    lucid.manual_seed(1)
    first = EPOCHS[name](lucid.Generator(11))()
    lucid.manual_seed(2)
    assert EPOCHS[name](lucid.Generator(11))() == first
    assert EPOCHS[name](lucid.Generator(12))() != first


@pytest.mark.parametrize("name", sorted(EPOCHS))
def test_a_generator_advances_from_one_epoch_to_the_next(name):
    epoch = EPOCHS[name](lucid.Generator(3))
    assert epoch() != epoch()


@pytest.mark.parametrize("name", sorted(EPOCHS))
def test_manual_seed_reproduces_the_default(name):
    lucid.manual_seed(5)
    epoch = EPOCHS[name](None)
    first, second = epoch(), epoch()
    assert first != second
    lucid.manual_seed(5)
    epoch = EPOCHS[name](None)
    assert [epoch(), epoch()] == [first, second]


@pytest.mark.parametrize("name", sorted(EPOCHS))
def test_an_int_seed_is_a_private_generator(name):
    """Same int, same draws — whatever the global generator is doing."""
    lucid.manual_seed(1)
    first = EPOCHS[name](9)()
    lucid.manual_seed(2)
    assert EPOCHS[name](9)() == first


@pytest.mark.parametrize("name", sorted(EPOCHS))
def test_a_generator_that_is_not_one_is_refused(name):
    with pytest.raises(TypeError, match="generator"):
        EPOCHS[name](object())()


def test_an_int_seed_no_longer_freezes_the_epochs():
    sampler = WeightedRandomSampler([1.0] * 100, 50, generator=0)
    seen = set()
    for _ in range(5):
        seen |= set(sampler)
    assert len(seen) > 80  # one epoch repeated would be <= 50


# ── nothing draws from a process-global RNG ──────────────────────────────────

# ``random.Random(seed)`` is an explicitly seeded stream (DistributedSampler,
# RASampler, seed derivation); ``random.seed`` is how a worker seeds the
# stream its dataset may use.  Every other ``random.*`` / ``np.random.*``
# call reads state nothing in the loader controls.
_SEEDED = {"Random", "seed"}


def _global_rng_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "random",
            "numpy.random",
        ):
            yield node.lineno, f"from {node.module} import ..."
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        attr = node.func.attr
        if isinstance(owner, ast.Name) and owner.id == "random":
            if attr not in _SEEDED or (attr == "Random" and not node.args):
                yield node.lineno, f"random.{attr}"
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "random"
            and isinstance(owner.value, ast.Name)
            and owner.value.id in ("np", "numpy")
            and attr != "seed"
        ):
            yield node.lineno, f"np.random.{attr}"


def test_no_module_draws_from_a_process_global_rng():
    root = Path(data.__file__).parent
    found = [
        f"{path.name}:{line}: {what}"
        for path in sorted(root.glob("*.py"))
        for line, what in _global_rng_calls(path)
    ]
    assert not found, found
