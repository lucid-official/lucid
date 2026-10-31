"""The random factories take every call form the reference does (LCD-773).

``randint`` took only ``(low, high, size)``, ``rand`` / ``randn`` refused
``size=``, ``normal`` had no positional shape and no tensor overloads, and
``manual_seed`` returned ``None``.  The table holds each form the reference
accepts to the same result — shape, dtype, device — on CPU and Metal, the
forms it refuses to a refusal, and the forms that mean the same draw to the
same values under one seed.
"""

import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._factories._overloads import spellings_of


class _Form(NamedTuple):
    """A call, written once and run against either library."""

    label: str
    call: Callable[[ModuleType, str], object]


def _vec(m: ModuleType, dev: str) -> object:
    return m.zeros(3, device=dev)


def _forms() -> list[_Form]:
    return [
        _Form("rand(2, 3)", lambda m, d: m.rand(2, 3, device=d)),
        _Form("rand((2, 3))", lambda m, d: m.rand((2, 3), device=d)),
        _Form("rand([2, 3])", lambda m, d: m.rand([2, 3], device=d)),
        _Form("rand(())", lambda m, d: m.rand((), device=d)),
        _Form("rand(size=(2, 3))", lambda m, d: m.rand(size=(2, 3), device=d)),
        _Form("rand(size=[2, 3])", lambda m, d: m.rand(size=[2, 3], device=d)),
        _Form("rand(size=())", lambda m, d: m.rand(size=(), device=d)),
        _Form("rand(size=(2,), dtype)", lambda m, d: m.rand(size=(2,), dtype=m.float16, device=d)),
        _Form("randn(2, 3)", lambda m, d: m.randn(2, 3, device=d)),
        _Form("randn(size=(2, 3))", lambda m, d: m.randn(size=(2, 3), device=d)),
        _Form("randn(size=[4])", lambda m, d: m.randn(size=[4], device=d)),
        _Form("randint(5, (3,))", lambda m, d: m.randint(5, (3,), device=d)),
        _Form("randint(5, size=(3,))", lambda m, d: m.randint(5, size=(3,), device=d)),
        _Form("randint(high=5, size=(3,))", lambda m, d: m.randint(high=5, size=(3,), device=d)),
        _Form("randint(1, 5, (3,))", lambda m, d: m.randint(1, 5, (3,), device=d)),
        _Form("randint(1, 5, size=(3,))", lambda m, d: m.randint(1, 5, size=(3,), device=d)),
        _Form("randint(1, high=5, size=)", lambda m, d: m.randint(1, high=5, size=(3,), device=d)),
        _Form(
            "randint(low=1, high=5, size=)",
            lambda m, d: m.randint(low=1, high=5, size=(3,), device=d),
        ),
        _Form(
            "randint(5, (2, 2), dtype)", lambda m, d: m.randint(5, (2, 2), dtype=m.int32, device=d)
        ),
        _Form("normal(0., 1., (3,))", lambda m, d: m.normal(0.0, 1.0, (3,), device=d)),
        _Form("normal(0., 1., size=(3,))", lambda m, d: m.normal(0.0, 1.0, size=(3,), device=d)),
        _Form(
            "normal(mean=, std=, size=)",
            lambda m, d: m.normal(mean=0.0, std=1.0, size=(3,), device=d),
        ),
        _Form("normal(0, 1, (3,)) ints", lambda m, d: m.normal(0, 1, (3,), device=d)),
        _Form("normal(mt, st)", lambda m, d: m.normal(m.zeros(3, device=d), m.ones(3, device=d))),
        _Form("normal(mt, 1.0)", lambda m, d: m.normal(m.zeros(3, device=d), 1.0)),
        _Form("normal(mt, std=2.0)", lambda m, d: m.normal(m.zeros(3, device=d), std=2.0)),
        _Form("normal(mt)", lambda m, d: m.normal(m.zeros(3, device=d))),
        _Form("normal(0., st)", lambda m, d: m.normal(0.0, m.ones(3, device=d))),
        _Form("normal(mean=0., std=st)", lambda m, d: m.normal(mean=0.0, std=m.ones(3, device=d))),
        _Form(
            "normal(mt(3), st(2, 3)) broadcasts",
            lambda m, d: m.normal(m.zeros(3, device=d), m.ones(2, 3, device=d)),
        ),
        _Form(
            "normal(mt, st, generator=)",
            lambda m, d: m.normal(
                m.zeros(3, device=d), m.ones(3, device=d), generator=m.Generator(device=d)
            ),
        ),
        _Form("randint_like(x, 5)", lambda m, d: m.randint_like(_vec(m, d), 5)),
        _Form("randint_like(x, 1, 5)", lambda m, d: m.randint_like(_vec(m, d), 1, 5)),
        _Form("randint_like(x, high=5)", lambda m, d: m.randint_like(_vec(m, d), high=5)),
        _Form(
            "randint_like(x, low=1, high=5)", lambda m, d: m.randint_like(_vec(m, d), low=1, high=5)
        ),
        _Form("randint_like(x, 1, high=5)", lambda m, d: m.randint_like(_vec(m, d), 1, high=5)),
        _Form("rand_like(x)", lambda m, d: m.rand_like(_vec(m, d))),
        _Form("randn_like(x)", lambda m, d: m.randn_like(_vec(m, d))),
        _Form("randperm(5)", lambda m, d: m.randperm(5, device=d)),
        _Form("randperm(n=5)", lambda m, d: m.randperm(n=5, device=d)),
        _Form("bernoulli(x)", lambda m, d: m.bernoulli(_vec(m, d))),
        _Form("bernoulli(x, 0.5)", lambda m, d: m.bernoulli(_vec(m, d), 0.5)),
        _Form("bernoulli(x, p=0.5)", lambda m, d: m.bernoulli(_vec(m, d), p=0.5)),
        _Form("bernoulli(input=x)", lambda m, d: m.bernoulli(input=_vec(m, d))),
        _Form("multinomial(w, 2)", lambda m, d: m.multinomial(m.ones(3, device=d), 2)),
        _Form(
            "multinomial(w, num_samples=2, replacement=True)",
            lambda m, d: m.multinomial(m.ones(3, device=d), num_samples=2, replacement=True),
        ),
        _Form("multinomial(w, 2, True)", lambda m, d: m.multinomial(m.ones(3, device=d), 2, True)),
    ]


def _refusals() -> list[_Form]:
    zeros = lambda m, d: m.zeros(3, device=d)  # noqa: E731 - a table row, not a function to name
    return [
        _Form("rand(2, 3, size=(2, 3))", lambda m, d: m.rand(2, 3, size=(2, 3), device=d)),
        _Form("rand(size=3)", lambda m, d: m.rand(size=3, device=d)),
        _Form("rand(3.0)", lambda m, d: m.rand(3.0, device=d)),
        _Form("randn((2, 3), (4,))", lambda m, d: m.randn((2, 3), (4,), device=d)),
        _Form("randint(5)", lambda m, d: m.randint(5, device=d)),
        _Form("randint(low=1, size=(3,))", lambda m, d: m.randint(low=1, size=(3,), device=d)),
        _Form("randint(5, size=3)", lambda m, d: m.randint(5, size=3, device=d)),
        _Form("randint(1, 5, 3)", lambda m, d: m.randint(1, 5, 3, device=d)),
        _Form("randint(5.0, (3,))", lambda m, d: m.randint(5.0, (3,), device=d)),
        _Form("randint(1, 2, 3, 4)", lambda m, d: m.randint(1, 2, 3, 4, device=d)),
        _Form("randint(5, (3,), 1)", lambda m, d: m.randint(5, (3,), 1, device=d)),
        _Form("randint(5, 5, (3,))", lambda m, d: m.randint(5, 5, (3,), device=d)),
        _Form("normal(0., 1.)", lambda m, d: m.normal(0.0, 1.0, device=d)),
        _Form("normal(0., 1., 3)", lambda m, d: m.normal(0.0, 1.0, 3, device=d)),
        _Form("normal(mt, st, size=)", lambda m, d: m.normal(zeros(m, d), zeros(m, d), size=(3,))),
        _Form("normal(mt, 1.0, size=)", lambda m, d: m.normal(zeros(m, d), 1.0, size=(3,))),
        _Form(
            "normal(mt, st, dtype=)",
            lambda m, d: m.normal(zeros(m, d), zeros(m, d), dtype=m.float32),
        ),
        _Form("normal(mt(3), st(4))", lambda m, d: m.normal(zeros(m, d), m.ones(4, device=d))),
        _Form("normal(mt, -st)", lambda m, d: m.normal(zeros(m, d), -m.ones(3, device=d))),
        _Form(
            "normal(int mt, int st)",
            lambda m, d: m.normal(
                m.zeros(3, dtype=m.int64, device=d), m.ones(3, dtype=m.int64, device=d)
            ),
        ),
        _Form("randint_like(x, low=1)", lambda m, d: m.randint_like(zeros(m, d), low=1)),
        _Form("randint_like(x)", lambda m, d: m.randint_like(zeros(m, d))),
    ]


def _describe(out: object) -> tuple[tuple[int, ...], str]:
    shape = tuple(getattr(out, "shape"))
    return shape, str(getattr(out, "dtype")).rsplit(".", 1)[-1]


@pytest.mark.parity
@pytest.mark.parametrize("form", _forms(), ids=lambda f: f.label)
def test_every_form_the_reference_takes_gives_its_shape_and_dtype(
    ref: ModuleType, device: str, form: _Form
) -> None:
    theirs = form.call(ref, "cpu")
    ours = form.call(lucid, device)
    assert _describe(ours) == _describe(theirs)
    assert getattr(ours, "device").type == device


@pytest.mark.parity
@pytest.mark.parametrize("form", _refusals(), ids=lambda f: f.label)
def test_every_form_the_reference_refuses_is_refused(
    ref: ModuleType, device: str, form: _Form
) -> None:
    with pytest.raises((TypeError, RuntimeError, ValueError)) as theirs:
        form.call(ref, "cpu")
    with pytest.raises((TypeError, RuntimeError, ValueError)) as ours:
        form.call(lucid, device)
    # The reference is a TypeError where the call is not any overload, and
    # the same here; where the arguments fit one and its value is refused
    # (``5, 5``, a negative std) both are a RuntimeError.
    assert issubclass(ours.type, TypeError) == issubclass(theirs.type, TypeError)


#: Spellings of one draw: under one seed each gives the same values.
_SAME_DRAW = {
    "rand": [
        lambda m, d: m.rand(2, 3, device=d),
        lambda m, d: m.rand((2, 3), device=d),
        lambda m, d: m.rand([2, 3], device=d),
        lambda m, d: m.rand(size=(2, 3), device=d),
        lambda m, d: m.rand(size=[2, 3], device=d),
    ],
    "randn": [
        lambda m, d: m.randn(2, 3, device=d),
        lambda m, d: m.randn((2, 3), device=d),
        lambda m, d: m.randn(size=(2, 3), device=d),
    ],
    "randint": [
        lambda m, d: m.randint(0, 5, (6,), device=d),
        lambda m, d: m.randint(5, (6,), device=d),
        lambda m, d: m.randint(5, size=(6,), device=d),
        lambda m, d: m.randint(high=5, size=(6,), device=d),
        lambda m, d: m.randint(low=0, high=5, size=(6,), device=d),
        lambda m, d: m.randint(0, high=5, size=(6,), device=d),
    ],
    "normal": [
        lambda m, d: m.normal(1.0, 2.0, size=(6,), device=d),
        lambda m, d: m.normal(1.0, 2.0, (6,), device=d),
        lambda m, d: m.normal(mean=1.0, std=2.0, size=(6,), device=d),
        lambda m, d: m.normal(1, 2, (6,), device=d),
    ],
    "normal of tensors": [
        lambda m, d: m.normal(m.full((4,), 1.0, device=d), m.full((4,), 2.0, device=d)),
        lambda m, d: m.normal(m.full((4,), 1.0, device=d), 2.0),
        lambda m, d: m.normal(m.full((4,), 1.0, device=d), std=2.0),
        lambda m, d: m.normal(1.0, m.full((4,), 2.0, device=d)),
    ],
    "randint_like": [
        lambda m, d: m.randint_like(m.zeros(5, device=d), 7),
        lambda m, d: m.randint_like(m.zeros(5, device=d), high=7),
        lambda m, d: m.randint_like(m.zeros(5, device=d), 0, 7),
        lambda m, d: m.randint_like(m.zeros(5, device=d), low=0, high=7),
    ],
}


@pytest.mark.parametrize("group", sorted(_SAME_DRAW))
def test_the_spellings_of_one_draw_give_the_same_values_under_one_seed(
    device: str, group: str
) -> None:
    drawn = []
    for call in _SAME_DRAW[group]:
        _ = lucid.manual_seed(11)
        drawn.append(call(lucid, device).tolist())
    assert all(d == drawn[0] for d in drawn[1:])


def test_a_seed_still_fixes_the_draw_and_a_new_one_changes_it(device: str) -> None:
    _ = lucid.manual_seed(1)
    a = lucid.randint(100, (8,), device=device).tolist()
    _ = lucid.manual_seed(1)
    assert lucid.randint(0, 100, (8,), device=device).tolist() == a
    _ = lucid.manual_seed(2)
    assert lucid.randint(100, (8,), device=device).tolist() != a


@pytest.mark.parity
def test_manual_seed_returns_the_default_generator(ref: ModuleType) -> None:
    ours = lucid.manual_seed(3)
    theirs = ref.manual_seed(3)
    assert isinstance(ours, _C_engine.Generator)
    assert ours is lucid.manual_seed(4)  # the one default stream, not a copy
    assert ours.seed == 4 and lucid.initial_seed() == 4
    assert theirs.initial_seed() == 3 and ref.manual_seed(4) is theirs
    # What it returns is the stream the factories read: drawing from it by
    # name continues the seeded sequence.
    _ = lucid.manual_seed(5)
    first = lucid.rand(3, generator=lucid.manual_seed(5)).tolist()
    _ = lucid.manual_seed(5)
    assert lucid.rand(3).tolist() == first


def test_a_generator_manual_seed_still_returns_itself() -> None:
    g = lucid.Generator()
    assert g.manual_seed(2) is g


def test_the_tensor_overloads_broadcast_and_carry_no_gradient(device: str) -> None:
    mean = lucid.zeros(3, device=device, requires_grad=True)
    std = lucid.ones(2, 3, device=device, requires_grad=True)
    out = lucid.normal(mean, std)
    assert out.shape == (2, 3) and not out.requires_grad
    big = lucid.normal(lucid.zeros(20_000, device=device), 3.0)
    assert abs(float(big.mean())) < 0.1 and abs(float(big.std()) - 3.0) < 0.1
    lo = lucid.normal(
        lucid.full((5_000,), 10.0, device=device), lucid.full((5_000,), 0.0, device=device)
    )
    assert lo.tolist() == [10.0] * 5_000  # std 0 is allowed: every sample is its mean


def test_the_tensor_overloads_promote_the_dtype(device: str) -> None:
    half = lucid.zeros(3, dtype=lucid.float16, device=device)
    assert lucid.normal(half, 1.0).dtype == lucid.float16
    assert lucid.normal(half, lucid.ones(3, device=device)).dtype == lucid.float32


def test_the_refusal_names_the_spellings_it_expected() -> None:
    with pytest.raises(TypeError) as err:
        lucid.randint(5)
    message = str(err.value)
    assert "invalid combination of arguments" in message
    assert "randint(high: int, size: tuple of ints)" in message
    assert "randint(low: int, high: int, size: tuple of ints)" in message


#: Files whose example still has a bare ``manual_seed`` line, held by another
#: workspace when LCD-773 was written; the guard fails the day one is fixed,
#: so the list cannot go stale.
_PENDING = frozenset({"lucid/utils/data/dataset.py", "lucid/_tensor/_methods.py"})


def test_no_docstring_example_leaves_manual_seed_unbound() -> None:
    """``manual_seed`` returns the generator, so a bare example line prints it."""
    root = Path(lucid.__file__).parent
    bare = re.compile(r"^\s*>>> lucid\.manual_seed\(", re.MULTILINE)
    offenders = {
        path.relative_to(root.parent).as_posix()
        for path in root.rglob("*.py")
        if "test" not in path.relative_to(root).parts and bare.search(path.read_text("utf-8"))
    }
    assert offenders == _PENDING, (
        "bind the result in these examples (`>>> _ = lucid.manual_seed(0)`), "
        "and drop a fixed file from _PENDING: "
        f"{sorted(offenders ^ _PENDING)}"
    )


def test_the_factories_declare_their_spellings() -> None:
    """The stub's overloads follow from these; a factory with none is a regression."""
    declared = {"rand": 2, "randn": 2, "randint": 2, "normal": 5, "randint_like": 2}
    for name, count in declared.items():
        assert len(spellings_of(getattr(lucid, name))) == count, name
