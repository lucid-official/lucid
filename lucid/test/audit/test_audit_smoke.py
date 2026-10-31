"""The audit harness has to work, so it gets its own tests.

Deliberately not the full sweep — that is minutes long and its output is
a report rather than an assertion.  What is checked here is that the
machinery cannot rot silently: the surface enumerates, every axis runs
without raising, the statuses mean what the summary claims, and the two
discriminators that keep the FAIL list short still discriminate.

The last point matters most.  Without the truncation check a
finite-difference probe near a pole reports a defect in ``reciprocal``,
and without the vacuity check a softmax contracted against ``ones``
reports a pass having tested nothing.  Both have happened.
"""

import numpy as np
import pytest

import lucid
from lucid.test.audit import _axes, _probe, _specs, _surface
from lucid.test.audit._console import Console, _strip
from lucid.test.audit._result import Baseline, Finding, Report, Status

# ── surface ──────────────────────────────────────────────────────────────────


def test_surface_enumerates_and_excludes_the_model_zoo() -> None:
    symbols = _surface.enumerate_surface()
    assert len(symbols) > 500
    assert not any(s.qualname.startswith("lucid.models") for s in symbols)
    assert "lucid.models" in _surface.EXCLUDED


def test_the_surface_reaches_every_public_object() -> None:
    """The guard on the denominator, and the reason ``--coverage`` exists.

    The enumeration used to name its modules in a literal.  The literal
    named nineteen; the package has a hundred and thirty, so a quarter of
    the framework — every augmentation in ``utils.transforms``, all of
    ``nn.init``, ``optim.lr_scheduler``, ``utils.data`` — was outside the
    denominator, and the tool reported a coverage percentage over what
    was left as though that were the whole.  Nothing failed; the number
    was simply of the wrong thing.

    This compares the surface against a traversal that shares none of its
    logic, so the two can only agree by both being right.
    """
    reached = {id(s.obj) for s in _surface.enumerate_surface()}
    absent = sorted(name for oid, name in _surface.independent_walk().items() if oid not in reached)
    assert not absent, f"public objects outside the audit surface: {absent[:20]}"


def test_no_subsystem_is_declared_and_then_silently_empty() -> None:
    """``lucid.signal`` was in the subsystem table and contributed nothing.

    Everything it exports is a sub-module, the enumeration skipped those,
    and the result was a subsystem that appeared in ``--list-subsystems``
    with zero symbols under it — printed in the output for weeks without
    being read as a bug.  Twelve window functions were outside the audit.

    An empty key is only allowed when it is an ancestor of a populated
    one: ``utils`` holds nothing directly but ``utils.data`` and
    ``utils.transforms`` are underneath it, so it is a container, and
    anything added to ``lucid.utils`` later files there rather than
    drifting into the ``lucid`` catch-all.  An empty *leaf* is the
    ``signal`` bug and fails here.
    """
    populated = {s.subsystem for s in _surface.enumerate_surface()}
    orphans = []
    for key in _surface.SUBSYSTEMS:
        if key in populated:
            continue
        if any(other.startswith(key + ".") for other in populated):
            continue  # a namespace container, not a lost subsystem
        orphans.append(key)
    assert not orphans, f"declared, empty, and not a container: {orphans}"


def test_every_callable_symbol_gets_a_concrete_invocation() -> None:
    """A spec nobody wrote is still a spec if the signature supplies it.

    ``_specs`` is hand-written, and 446 of 827 callable symbols reached
    only its generic ``f(x)`` / ``f(x, y)`` floor — the same
    hand-maintained-list failure that held reach at 73.8%.  Reading the
    signature closes it: what remains is symbols that take no arguments,
    where calling with none is the whole of what can be checked.
    """
    from lucid.test.audit import _autospec

    symbols = _surface.enumerate_surface()
    ops = [s for s in symbols if s.kind in ("op", "method") and s.inert]
    without = []
    for symbol in ops:
        fn = _surface.resolve(symbol)
        if fn is None or _specs.has_spec(symbol.short, qualname=symbol.qualname):
            continue
        if next(_autospec.invocations(fn, symbol.short, "moderate"), None) is None:
            without.append(symbol.qualname)
    assert len(without) <= 4, f"{len(without)} symbols reach only the ladder: {without}"


def test_a_keyword_only_op_does_not_crash_the_gradient_axes() -> None:
    """``affine_matrix(*, cx, cy)`` has an empty ``args``.

    ``Call.base`` used to index into it and raise ``IndexError``, which
    escapes as a harness ERROR; the axes read ``TypeError`` as "nothing
    to differentiate here" and skip, which is the truthful answer.
    """
    call = _specs.Call([], {"cx": 0.0, "cy": 0.0}, -1, "keyword-only")
    with pytest.raises(TypeError):
        _ = call.base


def test_every_axis_has_something_to_ask() -> None:
    """An axis that applies to nothing is a question no one is being asked."""
    symbols = _surface.enumerate_surface()
    idle = [a.name for a in _axes.ALL_AXES if not any(a.applies(s) for s in symbols)]
    assert not idle, f"axes with no applicable symbol: {idle}"


def test_kind_follows_the_defining_module_not_the_re_export() -> None:
    """``lucid.save`` is written in ``lucid.serialization`` and re-exported.

    Enumerating shallowest-first finds it under ``lucid``; keying its kind
    on where it was *found* made it a plain op and took the serialization
    axis to zero applicable symbols.
    """
    symbols = {s.qualname: s for s in _surface.enumerate_surface()}
    saver = symbols.get("lucid.save")
    assert saver is not None
    assert saver.subsystem == "serialization"


def test_stateful_symbols_are_marked_not_dropped() -> None:
    """They count towards the denominator; they are simply never called.

    A first sweep called ``set_grad_enabled`` with a tensor and poisoned
    every op after it alphabetically — 278 of them.

    Looked up across the whole surface rather than under ``lucid``: the
    qualname is the spelling a user writes, but a symbol is filed under
    the subsystem that *defines* it, and this one is written in
    ``lucid.autograd`` and re-exported.
    """
    symbols = {s.qualname: s for s in _surface.enumerate_surface()}
    poisonous = symbols.get("lucid.set_grad_enabled")
    assert poisonous is not None, "must still be counted"
    assert not poisonous.inert, "must never be invoked"
    assert poisonous.subsystem == "autograd", "filed where it is defined"


def test_every_subsystem_key_resolves() -> None:
    symbols = _surface.enumerate_surface()
    seen = {s.subsystem for s in symbols}
    assert seen <= set(_surface.SUBSYSTEMS)
    assert "nn.functional" in seen and "lucid" in seen


# ── specs ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["conv2d", "max_pool2d", "adaptive_avg_pool2d", "batch_norm", "layer_norm", "embedding"]
)
def test_the_families_that_used_to_be_skipped_have_specs(name: str) -> None:
    """Every convolution, pooling and normalisation was unreachable before."""
    assert _specs.has_spec(name), name
    assert next(_specs.invocations(name, "moderate"), None) is not None


def test_invocations_always_offer_a_fallback() -> None:
    """An unknown name must still get the generic ladder, not an empty iterator."""
    assert next(_specs.invocations("a_name_that_does_not_exist", "moderate"), None) is not None


def test_call_replaces_only_the_primary_argument() -> None:
    call = next(_specs.invocations("conv2d", "moderate"))
    swapped = call.with_primary(np.zeros(_probe.to_numpy(call.args[call.primary]).shape))
    assert float(lucid.abs(swapped.args[swapped.primary]).max()) == 0.0
    assert len(swapped.args) == len(call.args)


# ── discriminators ───────────────────────────────────────────────────────────


def test_quadratic_shrink_separates_truncation_from_a_real_error() -> None:
    # Truncation: refining h by 10 cuts the disagreement by ~100.
    assert _probe.quadratic_shrink(2.6e-4, 2.6e-6)
    # A wrong formula does not move.
    assert not _probe.quadratic_shrink(1.0, 0.99)
    assert not _probe.quadratic_shrink(5.3e-4, 4.9e-4)


def test_covector_is_never_constant() -> None:
    """A constant seed makes softmax's gradient identically zero."""
    weights = _probe.covector(16)
    assert weights.std() > 0.1
    assert not np.allclose(weights, weights[0])


def test_vacuous_is_reported_for_a_gradient_that_cannot_fail() -> None:
    axis = _axes.GradientAxis()
    symbol = _surface.Symbol("lucid.zeros_like", "lucid", "op", lucid.zeros_like)
    finding = axis.run(symbol, _axes.Context(quick=True))
    assert finding.status in (Status.VACUOUS, Status.UNSUPPORTED, Status.SKIP)


# ── axes ─────────────────────────────────────────────────────────────────────

_AXIS_PROBES = [
    ("grad", "lucid.tanh", lucid.tanh),
    ("creategraph", "lucid.exp", lucid.exp),
    ("grad2", "lucid.sin", lucid.sin),
    ("nonfinite", "lucid.exp", lucid.exp),
    ("entry", "lucid.exp", lucid.exp),
    ("edge", "lucid.exp", lucid.exp),
]


@pytest.mark.parametrize("axis_name,qualname,obj", _AXIS_PROBES)
def test_axis_runs_and_returns_a_finding(axis_name: str, qualname: str, obj: object) -> None:
    axis = _axes.axis_by_name(axis_name)
    assert axis is not None
    finding = axis.run(_surface.Symbol(qualname, "lucid", "op", obj), _axes.Context(quick=True))
    assert isinstance(finding, Finding)
    assert finding.axis == axis_name
    assert finding.status is not Status.ERROR, finding.detail


def test_a_known_good_op_passes_the_gradient_axis() -> None:
    """Guard the instrument: if nothing can pass, nothing can fail either."""
    axis = _axes.GradientAxis()
    finding = axis.run(
        _surface.Symbol("lucid.tanh", "lucid", "op", lucid.tanh), _axes.Context(quick=True)
    )
    assert finding.status is Status.PASS, finding.detail


def test_axes_declare_which_kinds_they_apply_to() -> None:
    for axis in _axes.ALL_AXES:
        assert axis.name and axis.summary
        assert axis.kinds


# ── report ───────────────────────────────────────────────────────────────────


def test_report_coverage_counts_only_reached_symbols() -> None:
    report = Report(surface_total=4, axes_run=["grad"])
    report.add(Finding("grad", "a", Status.PASS))
    report.add(Finding("grad", "b", Status.SKIP))
    reached, total, frac = report.coverage()
    assert (reached, total) == (1, 4)
    assert frac == pytest.approx(0.25)


def test_cell_coverage_uses_the_symbol_times_axis_matrix() -> None:
    report = Report(surface_total=10, axes_run=["grad", "device"])
    report.add(Finding("grad", "a", Status.PASS))
    filled, cells, frac = report.cell_coverage()
    assert (filled, cells) == (1, 20)
    assert frac == pytest.approx(0.05)


def test_skip_does_not_count_as_coverage() -> None:
    assert not Status.SKIP.is_coverage
    assert not Status.UNSUPPORTED.is_coverage
    assert Status.PASS.is_coverage and Status.FAIL.is_coverage


def test_only_fail_is_a_defect() -> None:
    defects = [s for s in Status if s.is_defect]
    assert defects == [Status.FAIL]


def test_report_serialises(tmp_path: pytest.TempPathFactory) -> None:
    from pathlib import Path

    report = Report(surface_total=2, axes_run=["grad"])
    report.add(Finding("grad", "lucid.exp", Status.PASS, "rel 1e-11", {"rel": 1e-11}))
    payload = report.as_dict()
    assert payload["coverage"]["symbols_total"] == 2
    assert payload["findings"][0]["symbol"] == "lucid.exp"
    out = Path(str(tmp_path)) / "nested" / "audit-report.json"
    report.write_json(out)
    assert out.is_file()


_INPLACE_PEER_METHODS = (
    "add_",
    "addcdiv_",
    "addcmul_",
    "copy_",
    "div_",
    "floor_divide_",
    "fmod_",
    "lerp_",
    "logical_and_",
    "logical_or_",
    "logical_xor_",
    "masked_fill_",
    "masked_scatter_",
    "maximum_",
    "minimum_",
    "mul_",
    "pow_",
    "remainder_",
    "sub_",
)


def test_inplace_tensor_methods_resize_tensor_peers_and_index_put_runs() -> None:
    """LCD-536: all 19 in-place Tensor methods shrink tensor peers (including
    boolean masks and broadcast peers) on both empty and size-1 edge probes,
    and ``index_put`` / ``index_put_`` pass a tuple of index tensors."""
    edge = _axes.axis_by_name("edge")
    entry = _axes.axis_by_name("entry")
    assert edge is not None and entry is not None
    ctx = _axes.Context(quick=True)
    for name in _INPLACE_PEER_METHODS:
        sym = _surface.Symbol(f"Tensor.{name}", "tensor", "method", getattr(lucid.Tensor, name))
        finding = edge.run(sym, ctx)
        assert finding.status is Status.PASS, (name, finding)
        assert finding.detail == "empty and size-1 accepted", (name, finding)
    for qual, obj in (
        ("lucid.index_put", lucid.index_put),
        ("Tensor.index_put_", lucid.Tensor.index_put_),
    ):
        kind = "method" if qual.startswith("Tensor.") else "op"
        sub = "tensor" if kind == "method" else "lucid"
        sym = _surface.Symbol(qual, sub, kind, obj)
        finding = entry.run(sym, ctx)
        assert finding.status is Status.PASS, (qual, finding)


# ── baseline ─────────────────────────────────────────────────────────────────


def test_baseline_downgrades_an_accepted_defect() -> None:
    baseline = Baseline({"nonfinite::lucid.sign": "convention, reviewed"})
    got = baseline.apply(Finding("nonfinite", "lucid.sign", Status.FAIL, "returned 0"))
    assert got.status is Status.KNOWN
    assert "convention" in got.detail


def test_baseline_leaves_an_unlisted_defect_alone() -> None:
    baseline = Baseline({})
    got = baseline.apply(Finding("grad", "lucid.exp", Status.FAIL, "rel 1.0"))
    assert got.status is Status.FAIL


def test_the_shipped_baseline_explains_every_entry() -> None:
    """An unexplained entry is indistinguishable from a silenced bug."""
    from pathlib import Path

    baseline = Baseline.load(Path(__file__).with_name("known.json"))
    for key, reason in baseline.entries.items():
        assert len(reason) > 40, f"{key} needs a real reason, got {reason!r}"


# ── console ──────────────────────────────────────────────────────────────────


def test_console_degrades_without_a_terminal() -> None:
    console = Console(colour=False, width=80)
    assert console.paint("x", "red") == "x"
    assert not console.live


def test_bar_and_strip_agree_on_width() -> None:
    console = Console(colour=False, width=80)
    assert len(_strip(console.bar(0, 10, 20))) == 20
    assert len(_strip(console.bar(10, 10, 20))) == 20
    assert len(_strip(console.bar(3, 10, 20))) == 20


def test_table_renders_without_a_terminal() -> None:
    console = Console(colour=False, width=80)
    console.table(["a", "bb"], [["1", "2"]])


# ── reach ────────────────────────────────────────────────────────────────────


def test_every_symbol_has_at_least_one_axis() -> None:
    """The property the tool is for.

    Reach went 57.3% -> 92.2% -> 100% as ``Tensor.*`` was resolved
    properly, the subsystems with no axis got one, and the smoke floor
    was widened to every callable.  If this regresses, some part of the
    framework has silently left the audit.

    Declarations are the one exemption, and it is narrow by
    construction: a Protocol has no implementation, a metaclass builds
    classes rather than values, and an abstract base refuses
    instantiation.  There is no probe to write for any of them.  The
    exemption is asserted to *stay* narrow below, so it cannot quietly
    become the place ops go to avoid being audited.
    """
    symbols = _surface.enumerate_surface()
    stranded = [
        s.qualname
        for s in symbols
        if s.kind != "declaration" and not any(a.applies(s) for a in _axes.ALL_AXES)
    ]
    assert not stranded, f"{len(stranded)} symbols no axis reaches: {stranded[:10]}"


def test_the_declaration_exemption_stays_narrow() -> None:
    """The exemption above is only safe while it holds nothing runnable.

    Every member has to be a Protocol, a metaclass or an abstract base —
    re-derived here from the objects rather than read off the
    classification, so a symbol cannot be exempted by being mislabelled.
    """
    declarations = [s for s in _surface.enumerate_surface() if s.kind == "declaration"]
    assert declarations, "the classification produced nothing — it stopped working"

    for symbol in declarations:
        obj = symbol.obj
        protocol = bool(getattr(obj, "_is_protocol", False))
        try:
            metaclass = isinstance(obj, type) and issubclass(obj, type)
        except TypeError:
            metaclass = False
        abstract = bool(getattr(obj, "__abstractmethods__", frozenset()))
        assert protocol or metaclass or abstract, (
            f"{symbol.qualname} is exempt from every axis but is none of "
            "protocol / metaclass / abstract base"
        )


def test_tensor_methods_resolve_to_something_callable() -> None:
    """253 symbols had no axis until properties and functions were told apart."""
    methods = [s for s in _surface.enumerate_surface(["tensor"]) if s.inert]
    assert len(methods) > 200
    unresolved = [s.qualname for s in methods if _surface.resolve(s) is None]
    assert not unresolved, unresolved[:10]


def test_no_namespace_is_counted_as_a_symbol() -> None:
    """``nn.functional`` is a subsystem, not a member of ``nn``."""
    import types as _types

    symbols = _surface.enumerate_surface()
    namespaces = [s.qualname for s in symbols if isinstance(s.obj, _types.ModuleType)]
    assert not namespaces, namespaces


def test_subsystem_kinds_reach_their_dedicated_axis() -> None:
    by_name = {a.name: a for a in _axes.ALL_AXES}
    for subsystem, axis_name in (
        ("distributions", "distribution"),
        ("diffeq", "diffeq"),
        ("optim", "optim"),
        ("serialization", "serialize"),
    ):
        symbols = _surface.enumerate_surface([subsystem])
        assert symbols, subsystem
        assert any(by_name[axis_name].applies(s) for s in symbols), subsystem


# ── the axes' own red light ──────────────────────────────────────────────────


def test_every_mutant_is_caught_by_the_axis_that_claims_it() -> None:
    """A green axis is not evidence until it has been made to go red.

    Each mutant is the framework broken in exactly the way one axis
    exists to notice.  An axis that stays green under its own defect
    class is decoration, and this is the test that says so — it is how
    ``broadcast`` was found comparing shapes and never values, and how
    the tokenizer round trip was found passing a decoder that silently
    dropped a character.
    """
    from lucid.test.audit import _mutants

    blind = [
        f"{verdict.axis}: {verdict.why} -> {verdict.status}"
        for verdict in _mutants.verify()
        if not verdict.caught
    ]
    assert not blind, "axes that did not notice their own defect: " + "; ".join(blind)


def test_the_unproven_list_does_not_grow_silently() -> None:
    """Adding an axis without a mutant is allowed, and has to be visible.

    The number is a ceiling, not a target: lower it by writing a mutant,
    never by deleting an axis.
    """
    from lucid.test.audit import _mutants

    unproven = _mutants.unproven_axes()
    assert len(unproven) <= 4, (
        f"{len(unproven)} axes have no mutant and nothing shows they can fail: " f"{unproven}"
    )
    unexplained = [name for name in unproven if name not in _mutants.UNPROVEN_REASONS]
    assert not unexplained, (
        "an axis may go unproven, and not silently — say why in " f"UNPROVEN_REASONS: {unexplained}"
    )
