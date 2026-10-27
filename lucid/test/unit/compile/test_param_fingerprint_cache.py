"""A compiled step keys its executable without walking the parameters (LCD-489).

The cache key holds every parameter's dtype and device, so ``.half()`` or a
move to another device compiles again.  ``make_step`` walked the parameters
for it on every step — 0.7–0.8 ms a step on bert_base — where
``CompiledModule`` kept the answer.  Both now take it from one owner,
``ParamFingerprint``, which walks again only after a change: a tensor
replaced under its ``Parameter`` (``.to()``, ``.half()``, an ``assign=True``
load — on any module of the tree), or a ``setattr`` that moves the tree
version.  The matrix below makes each change and holds the fingerprint to a
fresh walk.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import make_step
from lucid.compile._core import signature as sig
from lucid.test._fixtures.devices import metal_available


def _model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture
def walks(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every parameter walk taken for a key, by the model walked."""
    seen: list[object] = []
    real = sig._param_fingerprint

    def counted(model: nn.Module) -> tuple[tuple[str, str], ...]:
        seen.append(model)
        return real(model)

    monkeypatch.setattr(sig, "_param_fingerprint", counted)
    return seen


@pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning")
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "metal",
            marks=pytest.mark.skipif(not metal_available(), reason="metal unavailable"),
        ),
    ],
)
def test_a_replayed_make_step_walks_no_parameters(device: str, walks: list[object]) -> None:
    # On the CPU the step runs eager, keyed all the same.
    model = _model().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1)
    step = make_step(model, F.mse_loss)
    x = lucid.randn(3, 4, device=device)
    t = lucid.randn(3, 2, device=device)
    step(x, t).backward()
    opt.step()
    assert walks, "the first step took no fingerprint at all"
    walks.clear()
    for _ in range(5):
        opt.zero_grad()
        step(x, t).backward()
        opt.step()
    assert walks == [], f"{len(walks)} parameter walks over 5 replayed steps"


@pytest.mark.usefixtures("inference")
@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_replayed_compiled_module_walks_no_parameters(walks: list[object]) -> None:
    model = _model().to("metal")
    compiled = lucid.compile(model)
    x = lucid.randn(3, 4, device="metal")
    compiled(x)
    walks.clear()
    for _ in range(5):
        compiled(x)
    assert walks == []


def _assign_load(model: nn.Module) -> None:
    model.load_state_dict(
        {k: v.detach().half() for k, v in model.state_dict().items()}, assign=True
    )


def _new_parameter(model: nn.Module) -> None:
    model[2].weight = nn.Parameter(lucid.zeros(2, 8, dtype=lucid.float16))


# Each change, and whether it changes the fingerprint.
_CHANGES: dict[str, tuple[Callable[[nn.Sequential], object], bool]] = {
    "half": (lambda m: m.half(), True),
    "a_submodule_half": (lambda m: m[2].half(), True),
    "assign_load": (_assign_load, True),
    "a_submodule_assign_load": (lambda m: _assign_load(m[0]), True),
    "setattr_parameter": (_new_parameter, True),
    "a_new_submodule": (lambda m: m.append(nn.Linear(2, 2).half()), True),
    "to_same_device": (lambda m: m.to("cpu"), False),
    "train_eval": (lambda m: m.eval(), False),
    "copy_load": (lambda m: m.load_state_dict(m.state_dict()), False),
}


@pytest.mark.parametrize("change", list(_CHANGES))
def test_the_fingerprint_follows_every_change(change: str, walks: list[object]) -> None:
    apply, changes = _CHANGES[change]
    model = _model()
    fingerprint = sig.ParamFingerprint(model)
    before = fingerprint.get()
    assert fingerprint.get() == before
    assert len(walks) == 1, "a second get walked again with nothing changed"
    apply(model)
    after = fingerprint.get()
    assert after == sig._param_fingerprint(model)
    assert (after != before) == changes


def test_a_copy_of_the_owner_fingerprints_the_copied_model() -> None:
    import copy
    import pickle

    model = _model()
    holder = {"model": model, "fp": sig.ParamFingerprint(model)}
    holder["fp"].get()
    for clone in (copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))):
        assert clone["fp"].model is clone["model"]
        clone["model"][0].half()
        assert clone["fp"].get() == sig._param_fingerprint(clone["model"])
    assert holder["fp"].get() == (("lucid.float32", "cpu"),)
