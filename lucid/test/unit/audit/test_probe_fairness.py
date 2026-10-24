"""Each audit cell asks the op its own question, not one the probe made up.

Between 2026-10-04 and 10-16 the checks part of CI failed on audit verdicts
that were about the probes rather than the framework:

* ``fmod`` / ``remainder`` / ``floor_divide`` were "missing" every integer
  dtype on the CPU.  The float companions truncated to a zero divisor, and
  the CPU refuses integer division by zero (LCD-159) where Metal answers.
* ``Tensor.clamp(self, *args)`` was handed ``x.clamp(x, y)``: every element
  on its lower bound, a corner, reported as a missing double backward.
* ``Tensor.unfold`` and ``F.unfold`` were compared as one op, which they are
  not, in the reference either.
* ``ChainedScheduler`` was handed a scheduler of another optimizer, which it
  now refuses as the reference does.
* ``HalfNormal``'s ``icdf`` round trip compared a clipped probability's
  quantile with the unclipped point.
* ``hessian`` / ``jacobian`` / ``vjp`` got a list where the annotation says
  ``tuple``, and ``register_kl`` got ``nn.Linear`` for a distribution class.
* ``sampling.py`` imported its private grid helpers under public names, so
  ``F.reflect_grid`` was audited as API and failed the gradient axis for
  reading only its input's shape.

And on 2026-10-24 (LCD-561), eleven defects and six regressions, none of
them Lucid's — it matched the reference bit for bit:

* ``lucid.bernoulli`` and ``Tensor.bernoulli`` were compared as two
  unseeded draws.
* ``Tensor.ge_`` and the other in-place comparisons were asked to carry a
  NaN because their float receiver could, though a comparison answers a
  truth value (``[nan, 1, -1, 0].ge_(0)`` is ``[0, 1, 0, 1]``).
* ``GeneralizedPareto`` (batch ``(3,)``) was handed a length-9 cdf grid.
* ``TransformedDistribution`` gained a subclass and was taken for a base.
* ``alpha_dropout`` defaults to ``training=False`` and was probed for its
  draw in eval mode; ``bernoulli``'s probabilities were moved out of
  ``[0, 1]`` by the dtype axis.
* ``index_copy``, ``put_`` and ``scatter_`` were handed repeated indices,
  whose write order is undefined, and the M4 chose another write than the M1.
"""

import functools
import inspect
import re

import numpy as np
import pytest

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import (
    Context,
    DtypeAxis,
    EdgeAxis,
    EntryPointAxis,
    GradientAxis,
    NonFiniteAxis,
    SecondGradientAxis,
)
from lucid.test.audit._axes_data import SchedulerAxis
from lucid.test.audit._axes_stability import DeterminismAxis, RankAxis
from lucid.test.audit._axes_subsystem import DistributionAxis, SmokeAxis
from lucid.test.audit._result import Status


@functools.cache
def _symbols() -> dict[str, _surface.Symbol]:
    return {s.qualname: s for s in _surface.enumerate_surface()}


def _symbol(qualname: str) -> _surface.Symbol:
    return _symbols()[qualname]


def _context() -> Context:
    return Context(metal=_probe.metal_available())


@pytest.mark.parametrize(
    "qualname",
    [
        "Tensor.floor_divide",
        "Tensor.fmod",
        "Tensor.remainder",
        "lucid.fmod",
        "lucid.remainder",
    ],
)
def test_a_zero_divisor_refusal_is_not_missing_dtype_support(qualname: str) -> None:
    finding = DtypeAxis().run(_symbol(qualname), _context())
    assert finding.status == Status.PASS, finding.detail


def test_only_a_division_by_zero_counts_as_a_value_refusal() -> None:
    with pytest.raises(RuntimeError) as caught:
        lucid.fmod(
            lucid.tensor([3], dtype=lucid.int32), lucid.tensor([0], dtype=lucid.int32)
        )
    assert _probe.refuses_a_value(caught.value)
    assert _probe.refuses_a_value(ZeroDivisionError("integer division by zero"))
    assert not _probe.refuses_a_value(TypeError("dtype not supported: int8"))


def test_a_variadic_methods_operands_are_not_its_receivers_draw() -> None:
    symbol = _symbol("Tensor.clamp")
    fn = _surface.resolve(symbol)
    call = next(iter(_specs.invocations(symbol.short, "moderate", symbol.qualname, fn)))
    receiver, *rest = (_probe.to_numpy(a) for a in call.args)
    assert rest, "Tensor.clamp(self, *args) got no bound to compare"
    for bound in rest:
        assert bound is not None and not (bound == receiver).any()


@pytest.mark.parametrize("qualname", ["Tensor.clamp", "Tensor.clip"])
def test_clamp_is_not_probed_on_its_corner(qualname: str) -> None:
    for axis in (GradientAxis(), SecondGradientAxis()):
        finding = axis.run(_symbol(qualname), _context())
        assert finding.status == Status.PASS, f"{axis.name}: {finding.detail}"


def test_ops_that_only_share_a_name_are_not_compared() -> None:
    for qualname, labels in (
        ("Tensor.unfold", {"method"}),
        ("F.unfold", {"functional"}),
    ):
        got = {label for label, _ in _surface.counterparts(_symbol(qualname))}
        assert got == labels, (qualname, got)


def test_a_chained_scheduler_is_handed_schedulers_of_its_own_optimizer() -> None:
    finding = SchedulerAxis().run(_symbol("lucid.optim.ChainedScheduler"), _context())
    assert finding.status == Status.PASS, finding.detail


def test_the_icdf_round_trip_compares_only_points_it_did_not_clip() -> None:
    finding = DistributionAxis().run(
        _symbol("lucid.distributions.HalfNormal"), _context()
    )
    assert finding.status == Status.PASS, finding.detail


@pytest.mark.parametrize(
    "qualname",
    [
        "lucid.autograd.hessian",
        "lucid.autograd.jacobian",
        "lucid.autograd.vjp",
        "lucid.distributions.register_kl",
    ],
)
def test_signature_derived_calls_use_the_values_the_annotation_names(
    qualname: str,
) -> None:
    finding = SmokeAxis().run(_symbol(qualname), _context())
    assert finding.status == Status.PASS, finding.detail


@pytest.mark.parametrize(
    "helper",
    ["reflect_grid", "check_options", "composite_grid_sample", "affine_grid_3d"],
)
def test_private_grid_helpers_are_not_audited_as_api(helper: str) -> None:
    assert f"F.{helper}" not in _symbols()


def test_split_is_probed_with_sizes_its_axis_takes() -> None:
    # Since LCD-490 split refuses sizes that do not sum to the axis; the
    # derived ``(2, 2)`` fitted only a 1-D probe, and the rank axis lost it.
    finding = RankAxis().run(_symbol("lucid.split"), _context())
    assert finding.status == Status.PASS, finding.detail


def _stochastic() -> list[str]:
    return sorted(
        q
        for q, s in _symbols().items()
        if "stochastic" in s.flags and s.kind in ("op", "method")
    )


def test_every_spelling_of_a_sampler_is_drawn_under_one_seed() -> None:
    compared = [
        q for q in _stochastic() if len(list(_surface.counterparts(_symbol(q)))) >= 2
    ]
    assert compared, "no sampler has two spellings — the case this guards is gone"
    for qualname in compared:
        finding = EntryPointAxis().run(_symbol(qualname), _context())
        assert finding.status == Status.PASS, f"{qualname}: {finding.detail}"


def test_the_in_place_twin_of_a_nan_consumer_is_exempt_with_it() -> None:
    axis = NonFiniteAxis()
    twins = [
        q
        for q, s in _symbols().items()
        if q.startswith("Tensor.")
        and s.short.endswith("_")
        and s.short.removesuffix("_") in axis._CONSUMERS
    ]
    assert {"Tensor.ge_", "Tensor.lt_", "Tensor.heaviside_"} <= set(twins)
    for qualname in twins:
        assert not axis.applies(_symbol(qualname)), qualname
    # Only those: the in-place twin of an op that carries NaN still has to.
    assert axis.applies(_symbol("Tensor.exp_"))


def test_a_batched_distributions_cdf_is_probed_member_by_member() -> None:
    normal = lucid.distributions.Normal(
        lucid.tensor([0.0, 1.0, -1.0], dtype=lucid.float64),
        lucid.tensor([1.0, 0.5, 2.0], dtype=lucid.float64),
    )
    problems, checked = DistributionAxis._check_cdf(normal)
    assert not problems and {"cdf", "icdf"} <= set(checked), (problems, checked)
    finding = DistributionAxis().run(
        _symbol("lucid.distributions.GeneralizedPareto"), _context()
    )
    assert finding.status == Status.PASS, finding.detail


def test_a_base_with_a_sampler_of_its_own_is_audited_as_a_distribution() -> None:
    axis = DistributionAxis()
    assert axis.applies(_symbol("lucid.distributions.TransformedDistribution"))
    for base in ("Distribution", "ExponentialFamily"):
        assert not axis.applies(_symbol(f"lucid.distributions.{base}")), base


def _defaults_to_not_drawing(qualname: str) -> bool:
    fn = _surface.resolve(_symbol(qualname))
    training = inspect.signature(fn).parameters.get("training")
    return training is not None and training.default is False


def test_a_sampler_that_draws_only_in_training_is_asked_to_draw() -> None:
    gated = [q for q in _stochastic() if _defaults_to_not_drawing(q)]
    assert {"F.alpha_dropout", "F.feature_alpha_dropout", "F.rrelu"} <= set(gated)
    for qualname in gated:
        finding = DeterminismAxis().run(_symbol(qualname), _context())
        assert finding.status == Status.PASS, f"{qualname}: {finding.detail}"
    # A mode switch on an op that never draws is not a sampler's.
    batch_norm = _surface.resolve(_symbol("F.batch_norm"))
    assert not _specs._draws_only_when_training("batch_norm", batch_norm)


def test_bernoullis_probabilities_stay_probabilities_across_dtypes() -> None:
    for qualname in ("lucid.bernoulli", "Tensor.bernoulli"):
        finding = DtypeAxis().run(_symbol(qualname), _context())
        assert finding.status == Status.PASS, f"{qualname}: {finding.detail}"


_OVERWRITE = re.compile(r"^(scatter|index_copy|put|index_put)_?$")


def _index_operands(call: _specs.Call) -> list[np.ndarray]:
    operands = [*call.args]
    operands += [a for arg in call.args if isinstance(arg, tuple) for a in arg]
    return [
        np.asarray(_probe.to_numpy(a))
        for a in operands
        if getattr(a, "dtype", None) in (lucid.int32, lucid.int64)
    ]


def test_an_overwrite_by_index_names_each_position_once() -> None:
    overwrites = [q for q, s in _symbols().items() if _OVERWRITE.match(s.short)]
    assert {"Tensor.index_copy_", "Tensor.put_", "Tensor.scatter_"} <= set(overwrites)
    for qualname in overwrites:
        symbol = _symbol(qualname)
        fn = _surface.resolve(symbol)
        call, _, why = EdgeAxis()._working_call(fn, symbol, _context())
        assert call is not None, f"{qualname}: {why}"
        indices = _index_operands(call)
        assert indices, f"{qualname}: no index operand in {call.note}"
        for index in indices:
            for row in index.reshape(-1, index.shape[-1]):
                assert len(set(row.tolist())) == row.size, f"{qualname}: {row}"
        # The rest of the receiver survives, so the gradient and NaN axes
        # still have something to see.
        finding = NonFiniteAxis().run(symbol, _context())
        assert finding.status == Status.PASS, f"{qualname}: {finding.detail}"
