"""Every in-place method has its out-of-place twin (LCD-547).

``x.expm1_()`` existed and ``x.expm1()`` raised ``AttributeError``; so did
36 more — ``ldexp``, ``put``, ``xlogy``, ``greater`` … — each with a
``lucid.<name>`` free function and a reference method of the same name.
A registry op takes its method from its entry; the composites are
``lucid._tensor._free_fn_methods``'s table, each ``lucid.<name>`` as a
method.  Held here: the method is the free function (same values, same
refusals, on either device), a table method's signature is the free
function's after the first parameter, the reference has it and binds the
same keywords to the same values, and the class as a whole — no public
``X_`` without its ``X``.

LCD-633 widened the table past the in-place twins: 38 free functions
(``equal``, ``stft``, ``mv``, ``amax`` …), and the registry op ``conj``,
whose reference method Lucid's ``Tensor`` lacked.  Held for that class
too: every reference method whose free function ``lucid`` publishes is a
``Tensor`` method.
"""

import inspect
import math
from types import ModuleType

import pytest

import lucid
from lucid._ops._registry import _REGISTRY
from lucid._tensor._free_fn_methods import FREE_FN_METHODS


def _t(data: object, dtype: str = "float32") -> tuple[str, object, str]:
    return ("tensor", data, dtype)


_NAN = float("nan")
_INF = float("inf")


#: ``name -> (self, positional args, keyword args)`` for one representative
#: call, each tensor spelled ``_t(data, dtype)`` so either framework builds it.
_CASES: dict[str, tuple[object, tuple[object, ...], dict[str, object]]] = {
    "absolute": (_t([-1.5, 2.0]), (), {}),
    "adjoint": (_t([[1.0, 2.0], [3.0, 4.0]]), (), {}),
    "allclose": (_t([1.0, 2.0]), (_t([1.0, 2.000001]),), {"rtol": 1e-5}),
    "amax": (_t([[1.0, 5.0], [3.0, 2.0]]), (), {"dim": 1, "keepdim": True}),
    "amin": (_t([[1.0, 5.0], [3.0, 2.0]]), (), {"dim": 1, "keepdim": True}),
    "angle": (_t([1 + 1j, -1j], "complex64"), (), {}),
    "argwhere": (_t([[0.0, 1.0], [2.0, 0.0]]), (), {}),
    "conj": (_t([1 + 2j, -3j], "complex64"), (), {}),
    "corrcoef": (_t([[1.0, 2.0, 4.0], [2.0, 1.0, 0.5]]), (), {}),
    "count_nonzero": (_t([[0.0, 1.0], [2.0, 0.0]]), (), {"dim": 0}),
    "cov": (_t([[1.0, 2.0, 4.0], [2.0, 1.0, 0.5]]), (), {"correction": 0}),
    "diag_embed": (_t([[1.0, 2.0], [3.0, 4.0]]), (), {"offset": 1}),
    "equal": (_t([1.0, 2.0]), (_t([1, 2], "int64"),), {}),
    "fmax": (_t([1.0, _NAN, 3.0]), (_t([2.0, 1.0, _NAN]),), {}),
    "fmin": (_t([1.0, _NAN, 3.0]), (_t([2.0, 1.0, _NAN]),), {}),
    "ger": (_t([1.0, 2.0]), (_t([3.0, 4.0, 5.0]),), {}),
    "histogram": (_t([0.5, 1.5, 1.7, 3.0]), (4,), {"range": (0.0, 4.0)}),
    "is_conj": (_t([1.0]), (), {}),
    "is_neg": (_t([1.0]), (), {}),
    "is_nonzero": (_t([3.0]), (), {}),
    "is_same_size": (_t([1.0, 2.0]), (_t([[3.0, 4.0]]),), {}),
    "is_signed": (_t([1, 2], "int64"), (), {}),
    "isneginf": (_t([-_INF, 1.0, _INF]), (), {}),
    "isposinf": (_t([-_INF, 1.0, _INF]), (), {}),
    "isreal": (_t([1 + 0j, 1j], "complex64"), (), {}),
    # A two-bin spectrum of the frames [1, 2], [2, 3], [3, 4]: istft gives
    # the signal back.
    "istft": (
        _t([[3.0, 5.0, 7.0], [-1.0, -1.0, -1.0]], "complex64"),
        (2, 1, 2, _t([1.0, 1.0])),
        {"center": False},
    ),
    "logdet": (_t([[2.0, 1.0], [1.0, 3.0]]), (), {}),
    "moveaxis": (_t([[[1.0, 2.0]]]), (0, 2), {}),
    # One nonzero weight: the draw is the same in either framework.
    "multinomial": (_t([0.0, 1.0, 0.0]), (1,), {}),
    "mv": (_t([[1.0, 2.0], [3.0, 4.0]]), (_t([1.0, -1.0]),), {}),
    "nanmean": (_t([[1.0, _NAN], [3.0, 4.0]]), (1,), {"keepdim": True}),
    "nansum": (_t([[1.0, _NAN], [3.0, 4.0]]), (1,), {"keepdim": True}),
    "positive": (_t([1.0, -2.0]), (), {}),
    "resolve_conj": (_t([1.0, -2.0]), (), {}),
    "resolve_neg": (_t([1.0, -2.0]), (), {}),
    "rot90": (_t([[1.0, 2.0], [3.0, 4.0]]), (1, (0, 1)), {}),
    "signbit": (_t([-1.0, 0.0, 2.0]), (), {}),
    "stft": (
        _t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (4, 2, 4, _t([0.0, 0.5, 1.0, 0.5])),
        {"return_complex": True},
    ),
    "swapdims": (_t([[1.0, 2.0, 3.0]]), (0, 1), {}),
    "vdot": (_t([1.0, 2.0]), (_t([3.0, 4.0]),), {}),
    "acosh": (_t([1.5, 3.0]), (), {}),
    "addbmm": (
        _t([[0.5, 1.0, -1.0], [2.0, 0.0, 1.0]]),
        (
            _t([[[1.0, 2.0], [0.5, -1.0]]] * 2),
            _t([[[1.0, 0.0, 2.0], [3.0, 1.0, -1.0]]] * 2),
        ),
        {"beta": 0.5, "alpha": 2.0},
    ),
    "addmv": (
        _t([1.0, -2.0]),
        (_t([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]), _t([1.0, 0.5, -1.0])),
        {"beta": 2.0, "alpha": 0.5},
    ),
    "addr": (
        _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (_t([1.0, -1.0]), _t([0.5, 2.0, 1.0])),
        {"beta": 0.5, "alpha": 2.0},
    ),
    "arccosh": (_t([1.5, 3.0]), (), {}),
    "arcsinh": (_t([-0.5, 2.0]), (), {}),
    "arctan2": (_t([1.0, -2.0]), (_t([0.5, 3.0]),), {}),
    "arctanh": (_t([-0.5, 0.25]), (), {}),
    "asinh": (_t([-0.5, 2.0]), (), {}),
    "atanh": (_t([-0.5, 0.25]), (), {}),
    "baddbmm": (
        _t([[[1.0, 0.0], [0.0, 1.0]]] * 2),
        (
            _t([[[1.0, 2.0, 0.5], [0.0, -1.0, 1.0]]] * 2),
            _t([[[1.0, 0.0], [2.0, 1.0], [0.5, 0.5]]] * 2),
        ),
        {"beta": 0.5, "alpha": 2.0},
    ),
    "conj_physical": (_t([1 + 2j, -3j], "complex64"), (), {}),
    "copysign": (_t([1.0, -2.0, 3.0]), (_t([-1.0, 1.0, -0.5]),), {}),
    "digamma": (_t([0.5, 2.5]), (), {}),
    "erfc": (_t([-0.5, 1.0]), (), {}),
    "expm1": (_t([-0.5, 1.0, 1e-6]), (), {}),
    "float_power": (_t([2.0, 3.0]), (_t([2.0, 0.5]),), {}),
    "frexp": (_t([8.0, 0.5, -3.0]), (), {}),
    "gcd": (_t([12, 18, -9], "int64"), (_t([8, 27, 6], "int64"),), {}),
    "greater": (_t([1.0, 2.0, 3.0]), (_t([2.0, 2.0, 2.0]),), {}),
    "greater_equal": (_t([1.0, 2.0, 3.0]), (_t([2.0, 2.0, 2.0]),), {}),
    "heaviside": (_t([-1.0, 0.0, 2.0]), (_t([0.5, 0.5, 0.5]),), {}),
    "i0": (_t([0.5, 1.5]), (), {}),
    "lcm": (_t([4, 6, -3], "int64"), (_t([6, 9, 5], "int64"),), {}),
    "ldexp": (_t([1.0, 2.0, -0.75]), (_t([2, 3, -1], "int64"),), {}),
    "less": (_t([1.0, 2.0, 3.0]), (_t([2.0, 2.0, 2.0]),), {}),
    "less_equal": (_t([1.0, 2.0, 3.0]), (_t([2.0, 2.0, 2.0]),), {}),
    "lgamma": (_t([0.5, 3.0]), (), {}),
    "logit": (_t([0.05, 0.25, 0.75]), (), {"eps": 0.1}),
    "multiply": (_t([1.0, 2.0]), (_t([3.0, -4.0]),), {}),
    "negative": (_t([1.0, -2.0]), (), {}),
    "not_equal": (_t([1.0, 2.0, 3.0]), (_t([2.0, 2.0, 2.0]),), {}),
    "put": (
        _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (_t([0, 5, 0], "int64"), _t([10.0, 20.0, 1.0])),
        {"accumulate": True},
    ),
    "sinc": (_t([0.5, 1.5, 0.0]), (), {}),
    "subtract": (_t([3.0, 4.0]), (_t([1.0, 1.0]),), {"alpha": 2.0}),
    "xlogy": (_t([0.0, 2.0]), (_t([3.0, 4.0]),), {}),
}


def _build(spec: object, mod: ModuleType, device: str | None) -> object:
    if isinstance(spec, tuple) and spec and spec[0] == "tensor":
        _, data, dtype = spec
        kwargs: dict[str, object] = {"dtype": getattr(mod, dtype)}
        if device is not None:
            kwargs["device"] = device
        return mod.tensor(data, **kwargs)
    return spec


def _call(mod: ModuleType, name: str, device: str | None, method: bool) -> object:
    self_spec, args, kwargs = _CASES[name]
    x = _build(self_spec, mod, device)
    built = [_build(a, mod, device) for a in args]
    if method:
        return getattr(x, name)(*built, **kwargs)
    return getattr(mod, name)(x, *built, **kwargs)


def _parts(out: object) -> tuple[object, ...]:
    """A result's parts: ``frexp`` answers two tensors, ``equal`` a bool."""
    return tuple(out) if isinstance(out, tuple) else (out,)


def _values(out: object) -> list[object]:
    return [p.tolist() if hasattr(p, "tolist") else p for p in _parts(out)]


def _dtypes(out: object) -> list[str]:
    return [
        (
            str(getattr(p, "dtype")).rsplit(".", 1)[-1]
            if hasattr(p, "dtype")
            else type(p).__name__
        )
        for p in _parts(out)
    ]


def _close(a: object, b: object) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if not isinstance(a, int | float | complex) or not isinstance(
        b, int | float | complex
    ):
        return False
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(a) - complex(b)) <= 1e-5 * max(1.0, abs(complex(b)))
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


#: Twins a registry entry gives their method (``OpEntry.method_name``).
_REGISTRY_TWINS = (
    "conj",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
)


def test_every_twin_has_a_case_and_one_owner() -> None:
    assert sorted(_CASES) == sorted(FREE_FN_METHODS + _REGISTRY_TWINS)
    registry = {e.method_name for e in _REGISTRY}
    assert sorted(set(FREE_FN_METHODS) & registry) == []
    assert sorted(set(_REGISTRY_TWINS) - registry) == []
    for name in _CASES:
        assert callable(getattr(lucid, name)), name


@pytest.mark.parametrize("name", sorted(_CASES))
def test_the_method_is_the_free_function(name: str, device: str) -> None:
    try:
        expected = _call(lucid, name, device, method=False)
    except (NotImplementedError, RuntimeError, TypeError, ValueError) as err:
        # A refusal of the free function (Metal has no float64 for
        # ``float_power``) is the method's refusal too, word for word.
        with pytest.raises(type(err)) as got:
            _call(lucid, name, device, method=True)
        assert str(got.value) == str(err)
        return
    out = _call(lucid, name, device, method=True)
    assert _dtypes(out) == _dtypes(expected)
    assert _close(_values(out), _values(expected))


def _described(param: inspect.Parameter) -> tuple[object, ...]:
    return (param.name, param.kind, param.default, param.annotation)


@pytest.mark.parametrize("name", FREE_FN_METHODS)
def test_the_signature_is_the_free_functions_after_its_first(name: str) -> None:
    # Annotations included, evaluated: a snapshot taken before the package
    # bound its annotation names kept ``ForwardRef('Tensor')`` for good.
    method = inspect.signature(getattr(lucid.Tensor, name))
    free = inspect.signature(getattr(lucid, name))
    self_param, *rest = method.parameters.values()
    assert self_param.name == "self"
    assert [_described(p) for p in rest] == [
        _described(p) for p in list(free.parameters.values())[1:]
    ]
    assert method.return_annotation == free.return_annotation
    assert "ForwardRef" not in repr(method)


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(_CASES))
def test_the_reference_method_binds_our_keywords_to_our_values(
    ref: ModuleType, name: str
) -> None:
    """Every parameter passed by the name Lucid gives it: a renamed or
    missing keyword fails here, a different answer below."""
    assert hasattr(ref.Tensor, name)
    self_spec, args, kwargs = _CASES[name]
    params = list(inspect.signature(getattr(lucid, name)).parameters.values())[1:]
    named = {p.name: _build(a, ref, None) for p, a in zip(params, args, strict=False)}
    named.update(kwargs)
    theirs = getattr(_build(self_spec, ref, None), name)(**named)
    ours = _call(lucid, name, "cpu", method=True)
    assert _dtypes(ours) == _dtypes(theirs)
    assert _close(_values(ours), _values(theirs))


@pytest.mark.parity
@pytest.mark.xfail(
    strict=True,
    reason="LCD-545: signbit / copysign read the sign with `< 0`, so -0.0 and "
    "a negative NaN count as positive",
)
def test_copysign_takes_the_sign_bit_of_a_negative_zero(ref: ModuleType) -> None:
    ours = lucid.tensor([3.0, 3.0]).copysign(lucid.tensor([-0.0, -float("nan")]))
    theirs = ref.tensor([3.0, 3.0]).copysign(ref.tensor([-0.0, -float("nan")]))
    assert ours.tolist() == theirs.tolist() == [-3.0, -3.0]


#: In-place methods whose out-of-place form the reference lacks too, so
#: Lucid has none: ``fill_`` / ``zero_`` / ``copy_`` write a given value,
#: ``normal_`` / ``uniform_`` draw into the tensor, ``share_memory_`` moves
#: it — and ``cube_``, which the reference has in neither form.
_NO_TWIN = frozenset(
    {"copy_", "cube_", "fill_", "normal_", "share_memory_", "uniform_", "zero_"}
)


def _inplace_methods() -> list[str]:
    return sorted(
        n
        for n in dir(lucid.Tensor)
        if n.endswith("_")
        and not n.startswith("_")
        and callable(getattr(lucid.Tensor, n))
    )


def test_every_inplace_method_has_its_out_of_place_twin() -> None:
    missing = [
        n
        for n in _inplace_methods()
        if n not in _NO_TWIN and not hasattr(lucid.Tensor, n[:-1])
    ]
    assert missing == []
    # The list holds only names that are there to be excused.
    assert sorted(_NO_TWIN - set(_inplace_methods())) == []


@pytest.mark.parity
def test_the_excused_twins_are_ones_the_reference_lacks(ref: ModuleType) -> None:
    assert [n for n in sorted(_NO_TWIN) if hasattr(ref.Tensor, n[:-1])] == []


@pytest.mark.parity
@pytest.mark.xfail(
    strict=True,
    reason="LCD-545: signbit reads the sign with `< 0`, so -0.0 counts as positive",
)
def test_signbit_reads_the_sign_bit_of_a_negative_zero(ref: ModuleType) -> None:
    ours = lucid.tensor([-0.0, 0.0]).signbit()
    theirs = ref.tensor([-0.0, 0.0]).signbit()
    assert ours.tolist() == theirs.tolist() == [True, False]


#: Reference ``Tensor`` methods whose free function ``lucid`` publishes,
#: but which Lucid's ``Tensor`` has no method for, and why.
_NO_METHOD: dict[str, str] = {}


def _published_functions() -> set[str]:
    return {
        n
        for n in lucid.__all__
        if callable(getattr(lucid, n)) and not isinstance(getattr(lucid, n), type)
    }


@pytest.mark.parity
def test_every_reference_method_of_a_published_function_is_a_method(
    ref: ModuleType,
) -> None:
    lacking = sorted(
        n
        for n in _published_functions()
        if callable(getattr(ref.Tensor, n, None)) and not hasattr(lucid.Tensor, n)
    )
    assert [n for n in lacking if n not in _NO_METHOD] == []
    assert sorted(set(_NO_METHOD) - set(lacking)) == []


#: ``(input, other, dtypes, answer)`` as the reference answers ``equal``:
#: the shape exactly, the values at the promoted dtype, NaN equal to nothing.
_EQUAL_CASES: dict[str, tuple[object, object, tuple[str, str], bool]] = {
    "same values": ([1.0, 2.0], [1.0, 2.0], ("float32", "float32"), True),
    "a different value": ([1.0, 2.0], [1.0, 3.0], ("float32", "float32"), False),
    "int beside float": ([1, 2], [1.0, 2.0], ("int64", "float32"), True),
    "bool beside int": ([True, False], [1, 0], ("bool", "int64"), True),
    "complex beside float": ([1 + 0j], [1.0], ("complex64", "float32"), True),
    "nan": ([_NAN], [_NAN], ("float32", "float32"), False),
    "signed zero": ([-0.0], [0.0], ("float32", "float32"), True),
    "a longer other": ([1, 2], [1, 2, 3], ("int64", "int64"), False),
    "a broadcastable other": ([[1, 2]], [1, 2], ("int64", "int64"), False),
    "0-d beside one element": (1, [1], ("int64", "int64"), False),
    "both empty": ([], [], ("float32", "float32"), True),
}


def _dtype(mod: ModuleType, name: str) -> object:
    return getattr(mod, "bool_" if name == "bool" and mod is lucid else name)


@pytest.mark.parametrize("case", sorted(_EQUAL_CASES))
def test_equal_answers_one_bool_for_the_whole_pair(case: str, device: str) -> None:
    a, b, (da, db), want = _EQUAL_CASES[case]
    x = lucid.tensor(a, dtype=_dtype(lucid, da), device=device)
    y = lucid.tensor(b, dtype=_dtype(lucid, db), device=device)
    assert lucid.equal(x, y) is want
    assert x.equal(y) is want


@pytest.mark.parity
@pytest.mark.parametrize("case", sorted(_EQUAL_CASES))
def test_the_reference_answers_equal_the_same(ref: ModuleType, case: str) -> None:
    a, b, (da, db), want = _EQUAL_CASES[case]
    x = ref.tensor(a, dtype=_dtype(ref, da))
    y = ref.tensor(b, dtype=_dtype(ref, db))
    assert ref.equal(x, y) is want


@pytest.mark.parametrize("other", [1, 1.0, [1.0], None])
def test_equal_refuses_what_is_not_a_tensor(other: object) -> None:
    x = lucid.tensor([1.0])
    with pytest.raises(TypeError, match="lucid.equal"):
        lucid.equal(x, other)  # type: ignore[arg-type]  # the refusal under test
    with pytest.raises(TypeError, match="lucid.equal"):
        x.equal(other)


def test_isreal_is_false_where_the_imaginary_part_is_not_zero(device: str) -> None:
    x = lucid.tensor([1 + 0j, 1j, 2 - 3j, _NAN + 0j], device=device)
    assert lucid.isreal(x).tolist() == [True, False, False, True]
    assert x.isreal().tolist() == [True, False, False, True]
    assert lucid.isreal(lucid.tensor([1.0, _NAN], device=device)).tolist() == [
        True,
        True,
    ]
