"""Every in-place method has its out-of-place twin (LCD-547).

``x.expm1_()`` existed and ``x.expm1()`` raised ``AttributeError``; so did
36 more — ``ldexp``, ``put``, ``xlogy``, ``greater`` … — each with a
``lucid.<name>`` free function and a reference method of the same name.
``lucid._tensor._free_fn_methods`` makes each one ``lucid.<name>`` as a
method.  Held here: the method is the free function (same values, same
refusals, on either device), its signature is the free function's after the
first parameter, the reference has it and binds the same keywords to the
same values, and the class as a whole — no public ``X_`` without its ``X``.
"""

import inspect
import math
from types import ModuleType

import pytest

import lucid
from lucid._tensor._free_fn_methods import FREE_FN_METHODS


def _t(data: object, dtype: str = "float32") -> tuple[str, object, str]:
    return ("tensor", data, dtype)


#: ``name -> (self, positional args, keyword args)`` for one representative
#: call, each tensor spelled ``_t(data, dtype)`` so either framework builds it.
_CASES: dict[str, tuple[object, tuple[object, ...], dict[str, object]]] = {
    "absolute": (_t([-1.5, 2.0]), (), {}),
    "acosh": (_t([1.5, 3.0]), (), {}),
    "addbmm": (
        _t([[0.5, 1.0, -1.0], [2.0, 0.0, 1.0]]),
        (_t([[[1.0, 2.0], [0.5, -1.0]]] * 2), _t([[[1.0, 0.0, 2.0], [3.0, 1.0, -1.0]]] * 2)),
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
        (_t([[[1.0, 2.0, 0.5], [0.0, -1.0, 1.0]]] * 2), _t([[[1.0, 0.0], [2.0, 1.0], [0.5, 0.5]]] * 2)),
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
    """A result's tensors: ``frexp`` answers two."""
    return tuple(out) if isinstance(out, tuple) else (out,)


def _values(out: object) -> list[object]:
    return [getattr(p, "tolist")() for p in _parts(out)]


def _dtypes(out: object) -> list[str]:
    return [str(getattr(p, "dtype")).rsplit(".", 1)[-1] for p in _parts(out)]


def _close(a: object, b: object) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if not isinstance(a, int | float | complex) or not isinstance(b, int | float | complex):
        return False
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(a) - complex(b)) <= 1e-5 * max(1.0, abs(complex(b)))
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def test_every_table_name_has_a_case_and_a_free_function() -> None:
    assert sorted(_CASES) == sorted(FREE_FN_METHODS)
    for name in FREE_FN_METHODS:
        assert callable(getattr(lucid, name)), name


@pytest.mark.parametrize("name", FREE_FN_METHODS)
def test_the_method_is_the_free_function(name: str, device: str) -> None:
    try:
        expected = _call(lucid, name, device, method=False)
    except (RuntimeError, TypeError, ValueError) as err:
        # A refusal of the free function (Metal has no float64 for
        # ``float_power``) is the method's refusal too, word for word.
        with pytest.raises(type(err)) as got:
            _call(lucid, name, device, method=True)
        assert str(got.value) == str(err)
        return
    out = _call(lucid, name, device, method=True)
    assert _dtypes(out) == _dtypes(expected)
    assert _close(_values(out), _values(expected))


@pytest.mark.parametrize("name", FREE_FN_METHODS)
def test_the_signature_is_the_free_functions_after_its_first(name: str) -> None:
    method = inspect.signature(getattr(lucid.Tensor, name))
    free = inspect.signature(getattr(lucid, name))
    self_param, *rest = method.parameters.values()
    assert self_param.name == "self"
    assert [(p.name, p.kind, p.default) for p in rest] == [
        (p.name, p.kind, p.default) for p in list(free.parameters.values())[1:]
    ]


@pytest.mark.parity
@pytest.mark.parametrize("name", FREE_FN_METHODS)
def test_the_reference_method_binds_our_keywords_to_our_values(
    ref: ModuleType, name: str
) -> None:
    """Every parameter passed by the name Lucid gives it: a renamed or
    missing keyword fails here, a different answer below."""
    assert hasattr(ref.Tensor, name)
    self_spec, args, kwargs = _CASES[name]
    params = list(inspect.signature(getattr(lucid.Tensor, name)).parameters.values())[1:]
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
        if n.endswith("_") and not n.startswith("_") and callable(getattr(lucid.Tensor, n))
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
