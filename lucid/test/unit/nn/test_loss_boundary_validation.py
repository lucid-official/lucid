"""Every loss refuses a malformed argument at its boundary (CHA-233).

A loss used to compute first and look later.  A class ``weight`` of the
wrong length was cut short or read out of bounds — a silent answer on the
CPU, NaN or 0 on Metal; a BCE target of another shape was broadcast into a
different loss; ``multi_margin_loss`` raised its hinge to any power; and
nine losses looked at ``reduction`` only after computing.  Each public loss
now checks its arguments before its first op and refuses a bad one with a
typed exception, the same on the CPU and on Metal: a refusal dispatches no
op and waits on no device.

The losses are found by introspection — every function ``nn.functional``
exports with a ``reduction`` parameter, and every loss with a ``weight`` —
so a new loss fails :func:`test_every_loss_has_a_valid_call` until it has
one below, and from then on meets every row of the bad-argument table
that applies to it.  Under ``-m parity`` each row is checked against the
reference too: it refuses the same argument, except where a row is marked
as stricter.
"""

import ast
import inspect
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.nn.functional import loss as _loss_module
from lucid.nn.functional._loss_checks import _TARGET_SHAPE
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")

N, C = 4, 3

Kwargs = dict[str, object]


def losses_with(param: str) -> list[str]:
    """Every public ``nn.functional`` function with a parameter ``param`` —
    for ``weight``, only the losses (``linear`` and the norms have one too)."""
    found = []
    for name in F.__all__:
        fn = getattr(F, name, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if param not in params:
            continue
        if param == "weight" and fn.__module__ != _loss_module.__name__:
            continue
        found.append(name)
    return sorted(found)


LOSSES: list[str] = sorted(set(losses_with("reduction")) | set(losses_with("weight")))

#: How each loss's ``weight`` is shaped: one entry per class, a factor
#: that broadcasts against the elementwise loss, or the input's own shape.
#: A new weighted loss has to be put in one of the three.
CLASS_WEIGHT = frozenset({"cross_entropy", "nll_loss", "multi_margin_loss"})
ELEMENT_WEIGHT = frozenset(
    {"binary_cross_entropy", "binary_cross_entropy_with_logits", "multilabel_soft_margin_loss"}
)
SAME_WEIGHT = frozenset({"huber_loss", "l1_loss", "mse_loss"})


def _inputs(device: str) -> dict[str, lucid.Tensor]:
    lucid.manual_seed(0)
    return {
        "x": lucid.randn(N, C, device=device),
        "y": lucid.randn(N, C, device=device),
        "z": lucid.randn(N, C, device=device),
        "prob": lucid.rand(N, C, device=device) * 0.8 + 0.1,
        "binary": (lucid.rand(N, C, device=device) > 0.5).float(),
        "index": lucid.tensor([0, 2, 1, 0], device=device),
        "sign": lucid.tensor([1.0, -1.0, 1.0, -1.0], device=device),
        "s1": lucid.randn(N, device=device),
        "s2": lucid.randn(N, device=device),
        "labels": lucid.tensor([[0, -1, 0], [1, 2, -1], [2, -1, 1], [0, 1, 2]], device=device),
        "log_probs": F.log_softmax(lucid.randn(5, 2, C, device=device), dim=2),
        "ctc_targets": lucid.tensor([[1, 2], [2, 1]], device=device),
    }


#: An ordinary, valid call of each loss, as keyword arguments built from
#: :func:`_inputs`.
VALID: dict[str, Callable[[dict[str, lucid.Tensor]], Kwargs]] = {
    "binary_cross_entropy": lambda t: {"input": t["prob"], "target": t["binary"]},
    "binary_cross_entropy_with_logits": lambda t: {"input": t["x"], "target": t["binary"]},
    "cosine_embedding_loss": lambda t: {"input1": t["x"], "input2": t["y"], "target": t["sign"]},
    "cross_entropy": lambda t: {"input": t["x"], "target": t["index"]},
    "ctc_loss": lambda t: {
        "log_probs": t["log_probs"],
        "targets": t["ctc_targets"],
        "input_lengths": [5, 5],
        "target_lengths": [2, 2],
    },
    "gaussian_nll_loss": lambda t: {"input": t["x"], "target": t["y"], "var": t["prob"]},
    "hinge_embedding_loss": lambda t: {"input": t["s1"], "target": t["sign"]},
    "huber_loss": lambda t: {"input": t["x"], "target": t["y"]},
    "kl_div": lambda t: {"input": F.log_softmax(t["x"], dim=1), "target": F.softmax(t["y"], dim=1)},
    "l1_loss": lambda t: {"input": t["x"], "target": t["y"]},
    "margin_ranking_loss": lambda t: {"input1": t["s1"], "input2": t["s2"], "target": t["sign"]},
    "mse_loss": lambda t: {"input": t["x"], "target": t["y"]},
    "multi_margin_loss": lambda t: {"input": t["x"], "target": t["index"]},
    "multilabel_margin_loss": lambda t: {"input": t["x"], "target": t["labels"]},
    "multilabel_soft_margin_loss": lambda t: {"input": t["x"], "target": t["binary"]},
    "nll_loss": lambda t: {"input": F.log_softmax(t["x"], dim=1), "target": t["index"]},
    "poisson_nll_loss": lambda t: {"input": t["x"], "target": t["y"].abs()},
    "smooth_l1_loss": lambda t: {"input": t["x"], "target": t["y"]},
    "soft_margin_loss": lambda t: {"input": t["x"], "target": t["binary"] * 2.0 - 1.0},
    "triplet_margin_loss": lambda t: {"anchor": t["x"], "positive": t["y"], "negative": t["z"]},
    "triplet_margin_with_distance_loss": lambda t: {
        "anchor": t["x"],
        "positive": t["y"],
        "negative": t["z"],
    },
}


def valid_call(name: str, device: str) -> Kwargs:
    """Keyword arguments of an ordinary, valid call of loss ``name``."""
    return VALID[name](_inputs(device))


# ── the bad-argument table ────────────────────────────────────────────────


@dataclass(frozen=True)
class Bad:
    """One malformed argument: ``edit`` replaces fields of the valid call.

    ``stricter`` marks a row the reference accepts — Lucid refuses it on
    purpose, and the parity check skips it with that reason."""

    loss: str
    what: str
    edit: Callable[[str], Kwargs]
    exc: type[Exception] = ValueError
    match: str | None = None
    stricter: str = ""

    @property
    def id(self) -> str:
        return f"{self.loss}-{self.what}"


def _ones(*shape: int) -> Callable[[str], lucid.Tensor]:
    return lambda d: lucid.ones(*shape, device=d)


def _set(**fields: Callable[[str], object] | object) -> Callable[[str], Kwargs]:
    """An edit setting each field — to ``value(device)`` when callable."""

    def edit(device: str) -> Kwargs:
        return {k: v(device) if callable(v) else v for k, v in fields.items()}

    return edit


_BAD_REDUCTIONS: dict[str, object] = {
    "unknown": "avg",
    "none-object": None,
    "upper-case": "MEAN",
    "batchmean": "batchmean",
}


def _reduction_rows() -> list[Bad]:
    return [
        Bad(name, f"reduction-{key}", _set(reduction=value), match="reduction")
        for name in losses_with("reduction")
        for key, value in _BAD_REDUCTIONS.items()
        if not (key == "batchmean" and name == "kl_div")
    ]


def _weight_rows() -> list[Bad]:
    rows = []
    for name in sorted(CLASS_WEIGHT):
        rows += [
            Bad(name, "weight-too-long", _set(weight=_ones(C + 2)), match="weight"),
            Bad(name, "weight-too-short", _set(weight=_ones(C - 1)), match="weight"),
            Bad(name, "weight-2d", _set(weight=_ones(1, C)), match="weight"),
        ]
    for name in sorted(ELEMENT_WEIGHT):
        rows += [
            Bad(name, "weight-not-broadcast", _set(weight=_ones(C + 1)), match="weight"),
            Bad(
                name,
                "weight-enlarges",
                _set(weight=_ones(2, N, C)),
                match="weight",
                stricter=(
                    "the reference broadcasts the per-class loss up to the " "weight's shape"
                    if name == "multilabel_soft_margin_loss"
                    else ""
                ),
            ),
        ]
    for name in sorted(SAME_WEIGHT):
        rows += [
            Bad(name, "weight-wrong-shape", _set(weight=_ones(C)), match="weight"),
            Bad(name, "weight-enlarges", _set(weight=_ones(2, N, C)), match="weight"),
        ]
    return rows


def _class_target(d: str, *shape: int) -> lucid.Tensor:
    return lucid.zeros(*shape, dtype=lucid.int64, device=d)


_SPECIFIC: list[Bad] = [
    # class-index losses: the target is the input's shape minus the class dim
    *(
        row
        for name in ("cross_entropy", "nll_loss", "multi_margin_loss")
        for row in (
            Bad(
                name,
                "target-longer-batch",
                _set(target=lambda d: _class_target(d, N + 1)),
                match="target",
            ),
            Bad(
                name,
                "target-transposed",
                _set(target=lambda d: _class_target(d, 1, N)),
                match="target",
            ),
        )
    ),
    Bad(
        "cross_entropy",
        "input-0d",
        _set(input=lambda d: lucid.tensor(1.0, device=d), target=lambda d: _class_target(d)),
        match="input",
    ),
    Bad(
        "nll_loss",
        "input-0d",
        _set(input=lambda d: lucid.tensor(-1.0, device=d), target=lambda d: _class_target(d)),
        match="input",
    ),
    Bad(
        "nll_loss",
        "target-spatial-mismatch",
        _set(
            input=lambda d: lucid.randn(N, C, 2, device=d), target=lambda d: _class_target(d, N, 3)
        ),
        match="target",
    ),
    Bad(
        "multi_margin_loss",
        "input-3d",
        _set(input=lambda d: lucid.randn(N, C, 2, device=d)),
        match="input",
    ),
    *(Bad("multi_margin_loss", f"p-{p}", _set(p=p), match="p == 1") for p in (0, 3, -1)),
    *(
        Bad(
            "cross_entropy",
            f"label-smoothing-{tag}",
            _set(label_smoothing=value),
            match="label_smoothing",
            stricter=(
                "the reference checks only label_smoothing's upper bound, so "
                "it takes a negative or NaN one"
                if tag != "above-1"
                else ""
            ),
        )
        for tag, value in (("above-1", 1.5), ("negative", -0.1), ("nan", math.nan))
    ),
    *(
        Bad(
            name,
            "input-0d",
            _set(input=lambda d: lucid.tensor(0.5, device=d), target=lambda d: _class_target(d)),
            match="input",
            stricter="the reference takes a 0-d input: one class, a loss of 0",
        )
        for name in ("multi_margin_loss", "multilabel_margin_loss")
    ),
    Bad(
        "multilabel_margin_loss",
        "target-narrower",
        _set(target=lambda d: _class_target(d, N, C - 1)),
        match="target",
    ),
    Bad(
        "multilabel_margin_loss",
        "input-3d",
        _set(
            input=lambda d: lucid.randn(N, C, 2, device=d),
            target=lambda d: _class_target(d, N, C, 2),
        ),
        match="input",
    ),
    Bad(
        "binary_cross_entropy_with_logits",
        "pos-weight-not-broadcast",
        _set(pos_weight=_ones(C + 1)),
        match="pos_weight",
    ),
    Bad(
        "binary_cross_entropy_with_logits",
        "pos-weight-enlarges",
        _set(pos_weight=_ones(2, N, C)),
        match="pos_weight",
    ),
    # pair losses: one rank, shapes that line up
    Bad("cosine_embedding_loss", "target-2d", _set(target=_ones(N, 1)), match="target"),
    Bad(
        "cosine_embedding_loss",
        "target-0d-batched-inputs",
        _set(target=lambda d: lucid.tensor(1.0, device=d)),
        match="target",
    ),
    Bad("cosine_embedding_loss", "target-longer", _set(target=_ones(N + 1)), match="target"),
    Bad(
        "cosine_embedding_loss",
        "inputs-mismatch",
        _set(input2=lambda d: lucid.randn(N + 1, C, device=d)),
        match="inputs",
    ),
    Bad("margin_ranking_loss", "label-2d", _set(target=_ones(1, N)), match="dimensions"),
    Bad(
        "margin_ranking_loss",
        "scores-mismatch",
        _set(input2=lambda d: lucid.randn(N + 1, device=d)),
        match="dimensions",
    ),
    *(
        row
        for name in ("triplet_margin_loss", "triplet_margin_with_distance_loss")
        for row in (
            Bad(
                name,
                "negative-1d",
                _set(negative=lambda d: lucid.randn(C, device=d)),
                match="dimensions",
            ),
        )
    ),
    Bad(
        "triplet_margin_loss",
        "negative-longer",
        _set(negative=lambda d: lucid.randn(N + 1, C, device=d)),
        match="dimensions",
    ),
    # scalar arguments
    *(
        Bad("smooth_l1_loss", f"beta-{tag}", _set(beta=value), match="beta")
        for tag, value in (("negative", -1.0), ("nan", math.nan))
    ),
    *(
        Bad("huber_loss", f"delta-{tag}", _set(delta=value), match="delta")
        for tag, value in (("zero", 0.0), ("negative", -1.0))
    ),
    Bad("gaussian_nll_loss", "var-shape", _set(var=_ones(N + 1, C)), match="var"),
]


def _data_params(name: str) -> tuple[str, str]:
    """The names of loss ``name``'s input and target parameters."""
    first, second = list(inspect.signature(getattr(F, name)).parameters)[:2]
    return first, second


def _target_shaped(
    name: str, shape_of: Callable[[tuple[int, ...]], tuple[int, ...]]
) -> Callable[[str], Kwargs]:
    """An edit giving loss ``name`` a target of ``shape_of(input shape)``, at
    the valid target's dtype (an index stays an index)."""

    def edit(device: str) -> Kwargs:
        x_name, t_name = _data_params(name)
        call = valid_call(name, device)
        x, t = call[x_name], call[t_name]
        assert isinstance(x, lucid.Tensor) and isinstance(t, lucid.Tensor)
        shape = shape_of(tuple(x.shape))
        return {t_name: lucid.ones(*shape, dtype=t.dtype, device=device)}

    return edit


def _not_broadcast(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[:-1] + (shape[-1] + 1,)


def _one(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (1,)


def _enlarged(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (2, *shape)


def _target_rows() -> list[Bad]:
    """Rows from ``_TARGET_SHAPE`` itself, the loss module's one table of
    how each elementwise loss's target relates to its input."""
    rows = []
    for name, rule in sorted(_TARGET_SHAPE.items()):
        rows.append(
            Bad(name, "target-not-broadcast", _target_shaped(name, _not_broadcast), match="target")
        )
        if rule == "same":
            rows.append(
                Bad(name, "target-broadcastable", _target_shaped(name, _one), match="target size")
            )
        if rule == "within":
            rows.append(
                Bad(
                    name,
                    "target-enlarges",
                    _target_shaped(name, _enlarged),
                    match="target",
                    stricter=(
                        "the reference broadcasts the per-class loss up to the " "target's shape"
                        if name == "multilabel_soft_margin_loss"
                        else ""
                    ),
                )
            )
    return rows


BAD: list[Bad] = _reduction_rows() + _weight_rows() + _target_rows() + _SPECIFIC


def _call(row: Bad, device: str) -> Kwargs:
    kwargs = valid_call(row.loss, device)
    kwargs.update(row.edit(device))
    return kwargs


# ── the table's own invariants ────────────────────────────────────────────


def test_every_loss_has_a_valid_call() -> None:
    assert set(LOSSES) == set(VALID), "add a valid call for each new loss"


def test_every_weighted_loss_is_classified() -> None:
    assert set(losses_with("weight")) == CLASS_WEIGHT | ELEMENT_WEIGHT | SAME_WEIGHT


#: The losses whose target is not elementwise, and so not in
#: ``_TARGET_SHAPE``: one class index per sample, the pair losses' operands
#: (checked together), and ctc's label sequences.
CLASS_INDEX = frozenset({"cross_entropy", "nll_loss", "multi_margin_loss"})
PAIRED = frozenset(
    {
        "cosine_embedding_loss",
        "margin_ranking_loss",
        "triplet_margin_loss",
        "triplet_margin_with_distance_loss",
    }
)


def test_every_loss_has_one_target_rule() -> None:
    elementwise = set(_TARGET_SHAPE)
    assert elementwise | CLASS_INDEX | PAIRED | {"ctc_loss"} == set(LOSSES)
    assert not elementwise & (CLASS_INDEX | PAIRED)
    assert set(_TARGET_SHAPE.values()) <= {"same", "within", "broadcast", "broadcast-warn"}


def test_every_table_row_names_a_loss() -> None:
    assert {row.loss for row in BAD} <= set(LOSSES)
    assert len({row.id for row in BAD}) == len(BAD)


# ── the refusals ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("row", BAD, ids=[row.id for row in BAD])
def test_a_bad_argument_is_refused_before_any_op(row: Bad, device: str) -> None:
    kwargs = _call(row, device)
    fn = getattr(F, row.loss)
    syncs = _C_engine.host_sync_count()
    with lucid.profiler.profile() as prof:
        with pytest.raises(row.exc, match=row.match):
            fn(**kwargs)
    assert [event.name for event in prof.events()] == []
    assert _C_engine.host_sync_count() == syncs


@pytest.mark.parametrize("name", LOSSES)
def test_every_valid_call_still_runs(name: str, device: str) -> None:
    fn = getattr(F, name)
    reductions = ["none", "mean", "sum"] + (["batchmean"] if name == "kl_div" else [])
    for reduction in reductions:
        out = fn(**valid_call(name, device), reduction=reduction)
        assert out.device == device
        if reduction != "none":
            assert out.shape == ()


@pytest.mark.parametrize("name", sorted(CLASS_WEIGHT | ELEMENT_WEIGHT | SAME_WEIGHT))
def test_a_well_shaped_weight_is_taken(name: str, device: str) -> None:
    fn = getattr(F, name)
    shapes = (
        [(C,)]
        if name in CLASS_WEIGHT
        else [(N, C)] if name in SAME_WEIGHT else [(C,), (N, C), (N, 1), (1,), ()]
    )
    for shape in shapes:
        weight = lucid.full(shape, 2.0, device=device)
        out = fn(**valid_call(name, device), weight=weight, reduction="none")
        assert math.isfinite(float(out.sum().item()))


# ── binary_cross_entropy's value range: checked on the CPU only ───────────


_OUT_OF_RANGE = [("input", 1.5), ("input", -0.5), ("input", math.nan)] + [
    ("target", 1.5),
    ("target", -0.5),
    ("target", math.nan),
]


@pytest.mark.parametrize(("which", "value"), _OUT_OF_RANGE)
def test_bce_refuses_a_value_outside_0_1_on_the_cpu(which: str, value: float) -> None:
    p = lucid.tensor([0.2, 0.7, 0.5])
    y = lucid.tensor([0.0, 1.0, 1.0])
    if which == "input":
        p = lucid.tensor([0.2, value, 0.5])
    else:
        y = lucid.tensor([0.0, value, 1.0])
    with pytest.raises(ValueError, match=f"elements of {which}"):
        F.binary_cross_entropy(p, y)
    with pytest.raises(ValueError, match=f"elements of {which}"):
        nn.BCELoss()(p, y)


def test_bce_takes_the_ends_of_the_interval() -> None:
    p = lucid.tensor([0.0, 1.0, 0.0, 1.0])
    y = lucid.tensor([1.0, 0.0, 0.0, 1.0])
    assert F.binary_cross_entropy(p, y, reduction="none").tolist() == [100.0, 100.0, 0.0, 0.0]


@_needs_metal
def test_bce_on_metal_is_not_read_back() -> None:
    # A host read in every loss call would stall every training step; an
    # input outside [0, 1] makes the loss NaN there instead.
    p = lucid.tensor([0.2, 1.5, 0.5], device="metal")
    y = lucid.tensor([0.0, 1.0, 1.0], device="metal")
    syncs = _C_engine.host_sync_count()
    out = F.binary_cross_entropy(p, y, reduction="none")
    assert _C_engine.host_sync_count() == syncs
    got = out.tolist()
    assert math.isfinite(got[0]) and math.isnan(got[1]) and math.isfinite(got[2])


# ── what the boundary now accepts ─────────────────────────────────────────


_X = [[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]]


def test_label_smoothing_of_one_is_a_uniform_target(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    want = sum(
        -sum(v - math.log(sum(math.exp(u) for u in row)) for v in row) / 3 for row in _X
    ) / len(_X)
    got = F.cross_entropy(x, lucid.tensor([0, 1], device=device), label_smoothing=1.0)
    assert abs(got.item() - want) < 1e-5


def test_kl_div_batchmean_of_a_0d_input_is_its_sum(device: str) -> None:
    x = lucid.tensor(-1.0, device=device)
    t = lucid.tensor(0.5, device=device)
    batchmean = F.kl_div(x, t, reduction="batchmean")
    assert batchmean.shape == ()
    assert abs(batchmean.item() - F.kl_div(x, t, reduction="sum").item()) < 1e-7
    assert abs(batchmean.item() - 0.5 * (math.log(0.5) + 1.0)) < 1e-6


@pytest.mark.parametrize("label", [1.0, -1.0])
def test_cosine_embedding_takes_one_unbatched_pair(label: float, device: str) -> None:
    a = lucid.tensor([1.0, 0.0, 2.0], device=device)
    b = lucid.tensor([0.5, 0.5, -1.0], device=device)
    one = F.cosine_embedding_loss(
        a, b, lucid.tensor(label, device=device), margin=-0.5, reduction="none"
    )
    batched = F.cosine_embedding_loss(
        a.unsqueeze(0),
        b.unsqueeze(0),
        lucid.tensor([label], device=device),
        margin=-0.5,
        reduction="none",
    )
    assert one.shape == ()
    assert abs(one.item() - batched.tolist()[0]) < 1e-6


def test_cosine_embedding_broadcasts_a_batch_of_one_against_its_labels(device: str) -> None:
    # One pair against four labels is four losses, as on main and in the
    # reference; only shapes that do not broadcast at all are refused.
    a = lucid.randn(1, 3, device=device)
    b = lucid.randn(1, 3, device=device)
    labels = lucid.tensor([1.0, -1.0, 1.0, -1.0], device=device)
    out = F.cosine_embedding_loss(a, b, labels, reduction="none")
    assert out.shape == (4,)
    one = F.cosine_embedding_loss(a, b, labels[:1], reduction="none")
    assert abs(out.tolist()[0] - one.item()) < 1e-6


_BROADCAST_TARGETS = sorted(
    name for name, rule in _TARGET_SHAPE.items() if rule.startswith("broadcast")
)


@pytest.mark.parametrize("name", _BROADCAST_TARGETS)
def test_a_broadcast_target_is_the_explicit_broadcast(name: str, device: str) -> None:
    """A loss whose rule is ``broadcast`` takes a target that broadcasts —
    smaller than the input or larger — as the reference does; the fused
    ``mse_loss`` / ``huber_loss`` (and ``smooth_l1_loss`` over them)
    refused it with a bare ShapeMismatch.  ``broadcast-warn`` warns."""
    fn = getattr(F, name)
    x_name, t_name = _data_params(name)
    call = valid_call(name, device)
    x, t = call[x_name], call[t_name]
    assert isinstance(x, lucid.Tensor) and isinstance(t, lucid.Tensor)
    warns = _TARGET_SHAPE[name] == "broadcast-warn"

    one = t.reshape(-1)[:1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = fn(**{**call, t_name: one}, reduction="sum")
    assert any("target size" in str(w.message) for w in caught) == warns
    full = lucid.broadcast_to(one, list(x.shape))
    want = fn(**{**call, t_name: full}, reduction="sum")
    assert abs(got.item() - want.item()) <= 1e-5 * max(1.0, abs(want.item()))

    # A target that enlarges the loss: every input element is scored twice.
    xg = x.detach().requires_grad_()
    both = lucid.stack([t, t])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twice = fn(**{**call, x_name: xg, t_name: both}, reduction="sum")
        per = fn(**{**call, t_name: both}, reduction="none")
    assert per.shape == (2, *x.shape)
    twice.backward()
    xo = x.detach().requires_grad_()
    once = fn(**{**call, x_name: xo}, reduction="sum")
    once.backward()
    assert abs(twice.item() - 2 * once.item()) <= 1e-5 * max(1.0, abs(twice.item()))
    assert xg.grad is not None and xo.grad is not None
    assert xg.grad.shape == x.shape
    assert (xg.grad - 2 * xo.grad).abs().max().item() <= 1e-5


@_needs_metal
@pytest.mark.parametrize("n", [66000, 70000])
@pytest.mark.parametrize("half", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_under_autocast_past_65504_targets(
    n: int, half: lucid.dtype, reduction: str
) -> None:
    """Autocast gave the class-index mean's two sums the autocast dtype: a
    float16 count of 65504 kept samples is inf, and the mean was NaN with a
    gradient of 0 (a float16 sum, inf).  The loss's FP32 row, taken at its
    entry, runs them in float32 and answers in float32, as the reference's
    autocast does for a loss."""
    lucid.manual_seed(0)
    base = lucid.randn(n, 4, device="metal") * 0.1
    target = lucid.randint(0, 4, (n,), device="metal")
    x = base.detach().requires_grad_()
    with lucid.amp.autocast("metal", half):
        loss = F.cross_entropy(x, target, reduction=reduction)
    loss.backward()
    x32 = base.detach().requires_grad_()
    want = F.cross_entropy(x32, target, reduction=reduction)
    want.backward()
    assert loss.dtype == lucid.float32
    assert abs(loss.item() - want.item()) <= 1e-3 * abs(want.item())
    assert x.grad is not None and x32.grad is not None
    worst = (x.grad.float() - x32.grad).abs().max().item()
    assert worst <= 2e-2 * x32.grad.abs().max().item()


_AUTOCAST_SUMS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "l1_loss": lambda x: F.l1_loss(x, lucid.zeros_like(x), reduction="sum"),
    "smooth_l1_loss": lambda x: F.smooth_l1_loss(x, lucid.zeros_like(x), beta=0.5, reduction="sum"),
    "binary_cross_entropy_with_logits": lambda x: (
        F.binary_cross_entropy_with_logits(x, lucid.zeros_like(x), reduction="sum")
    ),
    "binary_cross_entropy": lambda x: F.binary_cross_entropy(
        lucid.sigmoid(x), lucid.zeros_like(x), reduction="sum"
    ),
}


@_needs_metal
@pytest.mark.parametrize("name", list(_AUTOCAST_SUMS))
def test_a_reduced_loss_under_autocast_is_float32(name: str) -> None:
    x = lucid.full((70000,), 2.0, device="metal")
    with lucid.amp.autocast("metal", lucid.float16):
        got = _AUTOCAST_SUMS[name](x)
    want = _AUTOCAST_SUMS[name](x)
    assert got.dtype == lucid.float32
    assert math.isfinite(got.item())
    assert abs(got.item() - want.item()) <= 1e-3 * abs(want.item())


@pytest.mark.parametrize(
    ("module", "args"),
    [
        (
            lambda: nn.CrossEntropyLoss(weight=lucid.ones(C + 2)),
            lambda: (lucid.randn(N, C), lucid.tensor([0, 1, 2, 0])),
        ),
        (lambda: nn.MultiMarginLoss(p=3), lambda: (lucid.randn(N, C), lucid.tensor([0, 1, 2, 0]))),
        (lambda: nn.BCEWithLogitsLoss(), lambda: (lucid.randn(N, C), lucid.ones(C))),
        (
            lambda: nn.KLDivLoss(reduction="avg"),  # type: ignore[arg-type]
            lambda: (lucid.randn(N, C), lucid.rand(N, C)),
        ),
    ],
    ids=["CrossEntropyLoss", "MultiMarginLoss", "BCEWithLogitsLoss", "KLDivLoss"],
)
def test_the_modules_inherit_the_refusals(
    module: Callable[[], nn.Module], args: Callable[[], tuple[lucid.Tensor, ...]]
) -> None:
    try:
        criterion = module()
    except ValueError:
        return  # refused at construction: as good
    with pytest.raises(ValueError):
        criterion(*args())


# ── one reduction helper ──────────────────────────────────────────────────


#: The functions allowed to reduce a whole tensor: the helper, and the
#: class-index losses' weighted mean (documented in ``_reduce``).
_REDUCERS = frozenset({"_reduce", "_weighted_mean"})


def _is_everything(node: ast.expr | None) -> bool:
    """A dimension argument that means "every dimension": absent, ``None``,
    or an empty list or tuple."""
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return node.value is None
    return isinstance(node, (ast.List, ast.Tuple)) and not node.elts


def _reduces_everything(call: ast.Call) -> bool:
    """Whether ``call`` is a ``mean`` / ``sum`` over a whole tensor: a
    method (``t.sum()``, ``t.sum(None)``, ``t.mean(dim=None)``), the free
    function (``lucid.mean(t)``, ``lucid.mean(t, dim=None)``) or the
    engine's (``_C_engine.sum(x, [], False)`` — but not its per-dim
    ``_C_engine.sum(x, [1], False)``)."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in ("mean", "sum"):
        return False
    base = func.value
    named = {kw.arg: kw.value for kw in call.keywords}
    dim_kw = named.get("dim", named.get("axis", named.get("axes")))
    if isinstance(base, ast.Name) and base.id in ("_C_engine", "_lucid", "lucid"):
        return _is_everything(call.args[1] if len(call.args) > 1 else dim_kw)
    return _is_everything(call.args[0] if call.args else dim_kw)


def _whole_reductions(tree: ast.Module) -> list[tuple[str, int]]:
    """``(enclosing top-level function, line)`` of every whole-tensor
    reduction in ``tree`` (:func:`_reduces_everything`)."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name if owner == "<module>" else owner
        if isinstance(node, ast.Call) and _reduces_everything(node):
            found.append((owner, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize(
    ("source", "whole"),
    [
        ("t.sum()", True),
        ("t.mean()", True),
        ("t.sum(None)", True),
        ("t.mean(dim=None)", True),
        ("t.sum(dim=())", True),
        ("_lucid.mean(t)", True),
        ("_lucid.mean(t, dim=None)", True),
        ("_C_engine.sum(x, [], False)", True),
        ("_C_engine.mean(x, None, False)", True),
        ("t.sum(dim=1)", False),
        ("t.mean(1)", False),
        ("t.sum(dim=[1, 2])", False),
        ("_lucid.mean(t, dim=-1, keepdim=False)", False),
        ("_C_engine.sum(x, [1], False)", False),
        ("t.all()", False),
    ],
)
def test_the_guard_tells_a_whole_reduction_from_a_per_dim_one(source: str, whole: bool) -> None:
    call = ast.parse(source, mode="eval").body
    assert isinstance(call, ast.Call)
    assert _reduces_everything(call) == whole


def test_one_helper_applies_every_reduction() -> None:
    tree = ast.parse(inspect.getsource(_loss_module))
    stray = [(fn, line) for fn, line in _whole_reductions(tree) if fn not in _REDUCERS]
    assert not stray, f"reduce through _reduce, not inline: {stray}"
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} | {
        target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign | ast.Assign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    retired = {"_REDUCTION_MAP", "_apply_reduction", "_reduce_in", "_validate_reduction"}
    assert not defined & retired


# ── against the reference ─────────────────────────────────────────────────


def _to_ref(R: object, value: object) -> object:
    if isinstance(value, lucid.Tensor):
        data = value.detach().to("cpu").tolist()
        if value.is_floating_point():
            return R.tensor(data, dtype=R.float32)  # type: ignore[attr-defined]
        return R.tensor(data, dtype=R.long)  # type: ignore[attr-defined]
    return value


#: How many leading parameters are the loss's data, passed positionally —
#: their names differ between the two (``x`` for ``input``, ``y`` for
#: ``target``); the options after them share their names.
_DATA_PARAMS: dict[str, int] = {
    "ctc_loss": 4,
    "cosine_embedding_loss": 3,
    "gaussian_nll_loss": 3,
    "margin_ranking_loss": 3,
    "triplet_margin_loss": 3,
    "triplet_margin_with_distance_loss": 3,
}


def _ref_call(R: object, name: str, kwargs: Kwargs) -> object:
    """Call the reference's loss ``name`` with the arguments of ours."""
    data = list(inspect.signature(getattr(F, name)).parameters)
    data = data[: _DATA_PARAMS.get(name, 2)]
    args = [_to_ref(R, kwargs[p]) for p in data]
    options = {p: _to_ref(R, v) for p, v in kwargs.items() if p not in data}
    return getattr(R.nn.functional, name)(*args, **options)  # type: ignore[attr-defined]


#: The reference's own advice (``kl_div``'s "mean", a broadcast regression
#: target) is not what these two tests ask about.
_QUIET_REFERENCE = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.mark.parity
@_QUIET_REFERENCE
@pytest.mark.parametrize("name", LOSSES)
def test_the_valid_calls_run_in_the_reference(ref: object, name: str) -> None:
    out = _ref_call(ref, name, valid_call(name, "cpu"))
    assert out is not None


@pytest.mark.parity
@_QUIET_REFERENCE
@pytest.mark.parametrize("row", BAD, ids=[row.id for row in BAD])
def test_the_reference_refuses_it_too(ref: object, row: Bad) -> None:
    if row.stricter:
        pytest.skip(f"Lucid is stricter here: {row.stricter}")
    with pytest.raises(Exception):  # noqa: B017 — its types differ from ours
        _ref_call(ref, row.loss, _call(row, "cpu"))


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(_TARGET_SHAPE))
def test_the_target_rules_are_the_references(ref: object, name: str) -> None:
    """Each loss's row of ``_TARGET_SHAPE`` is what the reference does: a
    one-element target is refused under ``same`` and taken otherwise (with
    a warning exactly under ``broadcast-warn``), and a target that enlarges
    the loss is taken only under ``broadcast*``."""
    rule = _TARGET_SHAPE[name]
    _, t_name = _data_params(name)
    call = valid_call(name, "cpu")
    t = call[t_name]
    assert isinstance(t, lucid.Tensor)

    def taken(target: lucid.Tensor) -> tuple[bool, bool]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _ref_call(ref, name, {**call, t_name: target})
            except Exception:  # noqa: BLE001 — its types differ from ours
                return False, False
        return True, any("target size" in str(w.message) for w in caught)

    ok, warned = taken(t.reshape(-1)[:1])
    assert ok == (rule != "same")
    assert warned == (rule == "broadcast-warn")
    enlarged, _ = taken(lucid.stack([t, t]))
    if name != "multilabel_soft_margin_loss":  # stricter by design
        assert enlarged == rule.startswith("broadcast")


@pytest.mark.parity
def test_the_accepted_cases_match_the_reference(ref: object, device: str) -> None:
    R = ref
    rf = R.nn.functional  # type: ignore[attr-defined]
    x, rx = lucid.tensor(_X, device=device), R.tensor(_X)  # type: ignore[attr-defined]
    w = [1.0, 2.0, 3.0]
    for target in ([0, 1], [0, -100]):
        for weight in (None, w):
            lo = F.cross_entropy(
                x,
                lucid.tensor(target, device=device),
                weight=None if weight is None else lucid.tensor(weight, device=device),
                label_smoothing=1.0,
            )
            ro = rf.cross_entropy(
                rx,
                R.tensor(target),  # type: ignore[attr-defined]
                weight=None if weight is None else R.tensor(weight),  # type: ignore[attr-defined]
                label_smoothing=1.0,
            )
            assert abs(lo.item() - ro.item()) < 1e-5
    lo = F.kl_div(
        lucid.tensor(-1.0, device=device), lucid.tensor(0.5, device=device), reduction="batchmean"
    )
    ro = rf.kl_div(R.tensor(-1.0), R.tensor(0.5), reduction="batchmean")  # type: ignore[attr-defined]
    assert abs(lo.item() - ro.item()) < 1e-6
    a, b = [1.0, 0.0, 2.0], [0.5, 0.5, -1.0]
    for label in (1.0, -1.0):
        lo = F.cosine_embedding_loss(
            lucid.tensor(a, device=device),
            lucid.tensor(b, device=device),
            lucid.tensor(label, device=device),
            margin=-0.5,
            reduction="none",
        )
        ro = rf.cosine_embedding_loss(
            R.tensor(a),  # type: ignore[attr-defined]
            R.tensor(b),  # type: ignore[attr-defined]
            R.tensor(label),  # type: ignore[attr-defined]
            margin=-0.5,
            reduction="none",
        )
        assert lo.shape == tuple(ro.shape) and abs(lo.item() - ro.item()) < 1e-5
    pa, pb, labels = [[0.3, -0.2, 0.9]], [[0.1, 0.5, 0.4]], [1.0, -1.0, 1.0, -1.0]
    lo = F.cosine_embedding_loss(
        lucid.tensor(pa, device=device),
        lucid.tensor(pb, device=device),
        lucid.tensor(labels, device=device),
        reduction="none",
    )
    ro = rf.cosine_embedding_loss(
        R.tensor(pa),  # type: ignore[attr-defined]
        R.tensor(pb),  # type: ignore[attr-defined]
        R.tensor(labels),  # type: ignore[attr-defined]
        reduction="none",
    )
    assert lo.shape == tuple(ro.shape)
    assert all(abs(a - b) < 1e-5 for a, b in zip(lo.tolist(), ro.tolist()))
