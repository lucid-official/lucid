"""``eval()`` twice on one input has to give one answer.

The zoo had no test for this. It is the only axis that sees a
regularisation layer which never learns it is at inference: a dropout
whose probability is forwarded unconditionally keeps dropping in
``eval()``, and nothing else in the suite notices. Shapes agree, the
training step still converges, published-weight parity still passes —
that comparison feeds tensors to sub-modules and reads the numbers once,
so a *stochastic* forward and a deterministic one are indistinguishable
to it unless you call twice.

V-JEPA 2 shipped with exactly that defect: ``dropout_p=attn_drop_rate``
straight into ``scaled_dot_product_attention``, which applies it with
``training=True`` regardless of the module's mode. It was invisible at
the default ``attn_drop_rate=0.0`` and would have surfaced as
"the same clip scores differently every call".

The families are the ones ``test_models_train_step.py`` already builds
at unit-test size, reused rather than re-declared: that table is the
zoo's list of "buildable through a public factory with a known small
config", and keeping one copy means a family added there is covered
here too.

Dropout is turned *on* where the config exposes it. With the default
0.0 the branch under test never runs, so a test at defaults would pass
against the broken code — which is why the original defect survived
every gate the zoo had.

The same holds one level down, for the graph a trace records (LCD-627).
A Core ML export, ``lucid.compile`` and ``make_step`` all record the
model's first traced call.  Swin built its shifted-window mask, and
CoAtNet its relative index, on first use inside the forward and kept it,
so a cold call recorded the ops that build the table and a warm one fed
the kept tensor: Swin's cold trace held 831 ops and its warm one 551.
The values agreed, so the eager check above could not see it.  Two
traces of one call must record one graph, and a table a family keeps
must reach it as a feed.
"""

import collections
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.models as models
from lucid.compile import _tracing
from lucid.models.vision.coatnet import _model as _coatnet
from lucid.test.unit.models.test_models_train_step import CASES, _family_of


def _values_of(case: Any) -> tuple[Any, ...]:
    """The case tuple, whether or not pytest.param wrapped it in marks."""
    return tuple(case.values) if hasattr(case, "values") else tuple(case)


def _tensors_of(output: Any) -> list[tuple[str, lucid.Tensor]]:
    """Every tensor a forward answered with, named for the message.

    The zoo's outputs are ``@dataclass(slots=True)``, which have no
    ``__dict__`` — ``vars()`` raises on some and returns nothing useful
    on others.  Detectors and a few backbones answer with a bare tensor
    or a tuple instead.
    """
    if isinstance(output, lucid.Tensor):
        return [("output", output)]
    if isinstance(output, (tuple, list)):
        return [(f"[{i}]", v) for i, v in enumerate(output) if isinstance(v, lucid.Tensor)]
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        found = []
        for field in dataclasses.fields(output):
            value = getattr(output, field.name, None)
            if isinstance(value, lucid.Tensor):
                found.append((field.name, value))
            elif isinstance(value, (tuple, list)):
                found += [
                    (f"{field.name}[{i}]", v)
                    for i, v in enumerate(value)
                    if isinstance(v, lucid.Tensor)
                ]
        return found
    return []


#: Config fields that put a dropout somewhere in the forward.  A family
#: that declares none is still worth running: the point is that *no*
#: source of non-determinism survives ``eval()``, and dropout is only
#: the one we have already been bitten by.
_DROPOUT_FIELDS = (
    "attn_drop_rate",
    "attention_dropout",
    "attention_probs_dropout_prob",
    "drop_rate",
    "dropout",
    "hidden_dropout_prob",
    "drop_path_rate",
)

#: Families whose forward is legitimately stochastic even at inference —
#: they draw, and the draw is the point.  Naming one here is a claim
#: that its randomness is the model's, not a leaked training-mode layer.
_SAMPLES_BY_DESIGN: dict[str, str] = {
    "vae": "reparameterised sampling is the forward",
    "hvae": "reparameterised sampling is the forward",
    "neural_ode": "the Hutchinson trace estimator draws a probe vector for log_det_jacobian",
    "tvm": "the forward is Algorithm 1's objective: it draws (t, s), w, the "
    "label drop and the noise",
}


def _with_dropout_on(factory: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Raise every dropout the factory's config declares off its default."""
    config = models.AutoConfig.from_pretrained(factory)
    enabled = dict(overrides)
    for field in _DROPOUT_FIELDS:
        if not hasattr(config, field):
            continue
        current = getattr(config, field)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            enabled[field] = 0.5
    return enabled


@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"), CASES, ids=[_family_of(c) for c in CASES]
)
def test_eval_is_deterministic(
    family: str, factory: str, overrides: dict[str, Any], make_inputs: Any
) -> None:
    """Two calls, one input, one answer — with dropout turned up."""
    # A model whose forward is a reparameterised draw is held to the same
    # seed on both calls: then the draw is the only randomness allowed.
    samples = family in _SAMPLES_BY_DESIGN

    lucid.manual_seed(0)
    model = models.create_model(factory, **_with_dropout_on(factory, overrides))
    model.eval()
    args, kwargs = make_inputs()

    with lucid.no_grad():
        if samples:
            lucid.manual_seed(1)
        first = model(*args, **kwargs)
        if samples:
            lucid.manual_seed(1)
        second = model(*args, **kwargs)

    left, right = _tensors_of(first), _tensors_of(second)
    assert left, f"{factory}: no tensor in {type(first).__name__} to compare"
    assert [n for n, _ in left] == [n for n, _ in right]
    for (name, a), (_, b) in zip(left, right):
        worst = float((a - b).abs().max().item())
        assert worst == 0.0, (
            f"{factory}.{name} differs between two eval() calls by {worst:g} — "
            "something in the forward is still drawing. A dropout handed its "
            "probability to a functional call without gating on self.training "
            "is the usual cause."
        )


def test_the_dropout_really_was_turned_on() -> None:
    """Guards the test above.

    At the zoo's default of 0.0 every dropout branch is skipped, so the
    determinism assertion would hold against code that never gates on
    training mode at all — which is how the defect this file exists for
    survived.  At least one case must actually raise a dropout.
    """
    raised = []
    for case in CASES:
        _family, factory, overrides, _make = _values_of(case)
        if _with_dropout_on(factory, overrides) != overrides:
            raised.append(factory)
    assert raised, "no case in CASES exposes a dropout field; this file proves nothing"


# -- the recorded graph ---------------------------------------------------------

#: Families whose kept tables outlive the model, by the call that drops them:
#: a fresh model alone does not make its first trace a cold one.
_SHARED_TABLES: dict[str, Callable[[], None]] = {"coatnet": _coatnet._INDEX_CACHE.clear}

#: Families that build a derived constant table on first use and keep it,
#: with the ops that build it.  None of them may enter a recording.
_TABLE_BUILDERS: dict[str, tuple[str, ...]] = {
    "coatnet": ("arange",),  # the relative-offset index
    "swin": ("arange", "scatter_set"),  # the shifted-window region mask
}

#: Families whose first traced call still records what a later one feeds,
#: each waiting on the card that may edit its model file — a strict xfail
#: with that card's ID as the reason.  Empty since TVM built its EMA networks
#: outside the trace (LCD-552); kept so the next such family has a home.
_COLD_TRACE_DIFFERS: dict[str, str] = {}


def _trace_cases() -> list[Any]:
    picked = []
    for case in CASES:
        family = _family_of(case)
        marks = list(getattr(case, "marks", ()))
        if family in _COLD_TRACE_DIFFERS:
            reason = _COLD_TRACE_DIFFERS[family]
            marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))
        picked.append(pytest.param(*_values_of(case), id=family, marks=marks))
    return picked


def _recorded(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    # Seeded alike, so a family that draws in its forward draws the same way
    # both times and any difference is the model's kept state.
    lucid.manual_seed(1)
    with _tracing(watch=[list(args), kwargs]) as tracer, lucid.no_grad():
        model(*args, **kwargs)
    return [op.name for op in tracer.graph.ops]


def _only_in(cold: list[str], warm: list[str]) -> str:
    extra = collections.Counter(cold) - collections.Counter(warm)
    return (
        f"the first trace recorded {len(cold)} ops and the second {len(warm)}; "
        f"only the first holds {dict(extra.most_common(8))}"
    )


@pytest.mark.parametrize(("family", "factory", "overrides", "make_inputs"), _trace_cases())
def test_a_trace_records_the_same_graph_cold_or_warm(
    family: str, factory: str, overrides: dict[str, Any], make_inputs: Any
) -> None:
    """A recording does not depend on what the model ran before (LCD-627)."""
    drop = _SHARED_TABLES.get(family)
    if drop is not None:
        drop()
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).eval()
    args, kwargs = make_inputs()

    cold = _recorded(model, args, kwargs)
    warm = _recorded(model, args, kwargs)

    assert cold == warm, f"{factory}: {_only_in(cold, warm)}"
    built = sorted(set(_TABLE_BUILDERS.get(family, ())) & set(cold))
    assert not built, f"{factory}: the trace builds a kept table with {built}"


def test_every_family_with_a_kept_table_is_traced() -> None:
    """Guards the test above: a name it lists must be a family it runs."""
    traced = {_family_of(case) for case in CASES}
    listed = set(_SHARED_TABLES) | set(_TABLE_BUILDERS) | set(_COLD_TRACE_DIFFERS)
    assert listed <= traced, f"not in CASES, so never traced: {listed - traced}"
