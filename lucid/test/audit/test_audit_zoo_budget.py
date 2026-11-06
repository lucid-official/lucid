"""The zoo's build budget holds a factory to a count whether or not it declares one (LCD-851).

The budget used to skip a factory by its registered ``params``, and about half
the zoo registers none, so ``genie`` (20 billion parameters, 75 GB as float32)
was built until the memory guard killed it.  The count is now measured: the
model is built under ``shadow_alloc``, which allocates nothing.  What is
pinned here is the decision that count feeds, with the registry and the
builder faked where a real build would be the thing being avoided, and the
guarantee under it, that a count exists for every factory the zoo registers.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import lucid
import lucid.metal
import lucid.models
from lucid.models import _registry, create_model
from lucid.nn._shadow import is_shadow, shadow_alloc
from lucid.test._default_device import default_device, metal_available
from lucid.test.audit import _zoo_budget, _zoo_probes
from lucid.test.audit._zoo_budget import BuildSize, MachineBudget, build_size, shadow_params
from lucid.test.audit._zoo_stage import CommandStage, read_result

#: The machines the budget is read against: the Studio, the MacBook, and the two
#: smaller sizes the budget's own table is written for.
_MACHINES = (36, 32, 16, 8)

#: ``genie`` registers four factories and none declares ``params``.
_GENIE_AT_PAPER_SIZE = ("genie", "genie_world_model")
_GENIE_COINRUN = ("genie_coinrun", "genie_coinrun_world_model")

#: Small real factories, for the checks that build a model for real.
_SMALL = ("lenet_5", "resnet_18")

_HUGE = 20_000_000_000
_HUGE_LINE = (
    "fake/huge: 20,000,000,000 params (counted without allocating) need 74.5 GB, "
    "past this machine's build budget of 22.4 GB"
)


# ── the count decides, for a factory that declares nothing ───────────────────


@pytest.mark.parametrize("gb", _MACHINES)
def test_genie_at_paper_size_is_past_every_machines_budget_and_coinrun_is_not(gb: int) -> None:
    """Two of ``genie``'s four factories are skipped; the other two are 0.7 GB and build."""
    budget = MachineBudget.for_memory(gb)
    for name in _GENIE_AT_PAPER_SIZE:
        size = build_size(name)
        assert size.source == "shadow" and size.params is not None
        assert not budget.fits_build(size.params), f"{name}: {size.params:,} params on {gb} GB"
    for name in _GENIE_COINRUN:
        size = build_size(name)
        assert size.source == "shadow" and size.params is not None
        assert budget.fits_build(size.params), f"{name}: {size.params:,} params on {gb} GB"


class _Param:
    def sum(self) -> _Param:
        return self

    def item(self) -> float:
        return 0.0

    def numel(self) -> int:
        return 6


class _Model:
    def parameters(self) -> list[_Param]:
        return [_Param()]


def _fake_zoo(
    monkeypatch: pytest.MonkeyPatch,
    counts: dict[str, int | None],
    declared: dict[str, int],
    built: list[str],
) -> None:
    """A family ``fake`` whose factories count ``counts`` (``None``: the count raises) and declare ``declared``.

    ``built`` collects the names the build probe really builds: a factory past the
    budget has to stay out of it, since building it is what the memory guard kills.
    """

    def count(name: str) -> int:
        size = counts[name]
        if size is None:
            raise RuntimeError("a phantom was read")
        return size

    def build(name: str) -> _Model:
        built.append(name)
        return _Model()

    monkeypatch.setattr(_zoo_budget, "shadow_params", count)
    monkeypatch.setattr(lucid.models, "create_model", build)
    monkeypatch.setattr(lucid.metal, "empty_cache", lambda: None)
    monkeypatch.setattr(
        _registry,
        "_REGISTRY",
        {name: SimpleNamespace(params=declared.get(name), factory=None) for name in counts},
    )
    monkeypatch.setattr(
        _zoo_probes,
        "factories_by_family",
        lambda: {"fake": _zoo_probes.Factories(names=list(counts), labels={"fake"})},
    )


def test_a_factory_with_no_declared_params_past_the_budget_is_skipped_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[str] = []
    _fake_zoo(monkeypatch, {"huge": _HUGE, "small": 6}, {}, built)
    assert _zoo_probes.build_family("fake", budget_gb=22.4) == 0
    out = capsys.readouterr().out
    assert built == ["small"], "the factory past the budget is never built"
    assert f"ADAPTIVE {_HUGE_LINE}" in out
    assert "1 built, 1 past budget" in out


def test_a_factory_under_the_budget_is_still_built(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[str] = []
    _fake_zoo(monkeypatch, {"a": 6, "b": 600_000_000}, {}, built)  # 2.2 GB of float32
    assert _zoo_probes.build_family("fake", budget_gb=22.4) == 0
    assert built == ["a", "b"]
    assert "ADAPTIVE" not in capsys.readouterr().out


def test_a_skip_by_count_is_a_note_and_a_reduced_family_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """What the family note and the verdict show: the reason, and ``reduced`` rather than ``mem-guard``."""
    _fake_zoo(monkeypatch, {"huge": _HUGE, "small": 6}, {}, [])
    code = _zoo_probes.build_family("fake", budget_gb=22.4)
    log = tmp_path / "build.log"
    log.write_text(capsys.readouterr().out)
    result = read_result(CommandStage(name="build", command=("x",)), log, "pass", 1.0, 0.1)
    assert code == 0 and not result.failed and result.reduced
    assert result.notes == [_HUGE_LINE]


# ── where the number comes from ──────────────────────────────────────────────


def test_a_measured_count_outranks_a_declared_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``params`` is the paper's figure; a factory can build well over what it cites."""
    monkeypatch.setattr(_zoo_budget, "shadow_params", lambda name: _HUGE)
    assert build_size("x", 1_000_000) == BuildSize(_HUGE, "shadow")


def test_the_declared_count_stands_in_only_when_the_count_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[str] = []
    _fake_zoo(monkeypatch, {"declares": None, "silent": None}, {"declares": _HUGE}, built)
    assert build_size("declares", _HUGE) == BuildSize(
        _HUGE, "declared", "RuntimeError: a phantom was read"
    )
    assert build_size("silent", None) == BuildSize(None, "none", "RuntimeError: a phantom was read")

    assert _zoo_probes.build_family("fake", budget_gb=22.4) == 0
    out = capsys.readouterr().out
    assert "ADAPTIVE fake/declares: 20,000,000,000 params (as declared) need 74.5 GB" in out
    assert built == ["silent"], "a factory with no count at all is built, as before"
    assert "CHECK silent: no parameter count to hold the build to the budget" in out
    assert "a phantom was read); built without one" in out


def test_only_the_errors_of_shadow_mode_are_taken_for_a_count_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(name: str) -> int:
        raise KeyError(name)

    monkeypatch.setattr(_zoo_budget, "shadow_params", broken)
    with pytest.raises(KeyError):
        build_size("x", 10)


# ── the count is the build's, and counting leaves nothing behind ─────────────


@pytest.mark.parametrize("name", _SMALL)
def test_the_shadow_count_is_what_a_real_build_holds(name: str) -> None:
    built = sum(int(param.numel()) for param in create_model(name).parameters())
    assert shadow_params(name) == built > 0


def test_counting_leaves_the_default_device_and_the_next_build_real() -> None:
    before = lucid.get_default_device()
    shadow_params(_SMALL[0])
    assert lucid.get_default_device() == before
    model = create_model(_SMALL[0])
    assert not any(is_shadow(param) for param in model.parameters())
    _zoo_probes.force_allocation(model)


@pytest.mark.skipif(not metal_available(), reason="the claim is about two devices")
@pytest.mark.parametrize("name", _SMALL)
def test_the_count_does_not_depend_on_the_default_device(name: str) -> None:
    """Why it is taken on the CPU: the device could only change whether counting works."""
    with default_device("metal"), shadow_alloc():
        model = create_model(name)
    on_metal = sum(int(param.numel()) for param in model.parameters())
    assert shadow_params(name) == on_metal


def test_every_registered_factory_has_a_count() -> None:
    """The class guard: a factory the count cannot reach is one the budget cannot skip.

    This builds every factory in the zoo without storage (about 10 seconds), so a
    new family whose construction a shadow block cannot trace fails here, in the
    unit tests, rather than as a ``mem-guard`` in the next whole-zoo run.
    """
    uncounted: dict[str, str] = {}
    for name, entry in _registry._REGISTRY.items():
        size = build_size(name, entry.params)
        if size.source != "shadow" or not size.params:
            uncounted[name] = size.note or "a model with no parameters"
    assert not uncounted, uncounted
