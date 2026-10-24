"""Every reference dtype-cast method is a ``Tensor`` method (LCD-633).

``x.half()`` and ``x.long()`` existed; ``x.bfloat16()``, ``x.short()``,
``x.char()``, ``x.cfloat()`` and ``x.cdouble()`` raised ``AttributeError``.
Each is ``x.to(<dtype>)``, and ``self`` when it is at that dtype already.
``byte()`` (uint8) and ``chalf()`` (complex32) name dtypes Lucid does not
have: they are refused, not answered at a different dtype.  Metal has no
float64 or complex128, so those casts are refused there as ``to`` refuses
them.
"""

import inspect
import re
from types import ModuleType

import pytest

import lucid

#: ``method -> dtype`` it casts to.
_CASTS: dict[str, str] = {
    "bfloat16": "bfloat16",
    "bool": "bool_",
    "cdouble": "complex128",
    "cfloat": "complex64",
    "char": "int8",
    "double": "float64",
    "float": "float32",
    "half": "float16",
    "int": "int32",
    "long": "int64",
    "short": "int16",
}

#: Reference cast methods to a dtype Lucid lacks: refused, ``NotImplementedError``.
_REFUSED: dict[str, str] = {"byte": "uint8", "chalf": "complex32"}

#: Dtypes Metal does not hold.
_CPU_ONLY = frozenset({"float64", "complex128"})


@pytest.mark.parametrize("name", sorted(_CASTS))
def test_the_cast_answers_its_dtype(name: str, device: str) -> None:
    target = getattr(lucid, _CASTS[name])
    x = lucid.tensor([1.5, -2.0, 0.0], device=device)
    if device == "metal" and _CASTS[name] in _CPU_ONLY:
        with pytest.raises(NotImplementedError, match=_CASTS[name]):
            getattr(x, name)()
        return
    out = getattr(x, name)()
    assert out.dtype == target
    assert out.device == x.device
    assert out.tolist() == x.to(target).tolist()
    # At the dtype already: the tensor itself, as ``to`` answers.
    assert getattr(out, name)() is out


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_a_cast_to_a_dtype_lucid_lacks_is_refused(name: str, device: str) -> None:
    x = lucid.ones(2, device=device)
    with pytest.raises(NotImplementedError, match=_REFUSED[name]):
        getattr(x, name)()


@pytest.mark.parametrize("name", sorted(_CASTS | _REFUSED))
def test_the_cast_takes_no_arguments(name: str) -> None:
    params = list(inspect.signature(getattr(lucid.Tensor, name)).parameters)
    assert params == ["self"]


def _reference_casts(ref: ModuleType) -> set[str]:
    """The reference's ``self.<m>()`` is equivalent to ``self.to(<dtype>)`` methods."""
    pattern = re.compile(r"``self\.(\w+)\(\)`` is equivalent to ``self\.to\(")
    found = set()
    for name in dir(ref.Tensor):
        doc = getattr(getattr(ref.Tensor, name, None), "__doc__", None) or ""
        match = pattern.search(doc)
        if match and match.group(1) == name:
            found.add(name)
    return found


@pytest.mark.parity
def test_every_reference_cast_method_is_here(ref: ModuleType) -> None:
    theirs = _reference_casts(ref)
    assert len(theirs) >= 10  # the docstring scan found them
    assert sorted(theirs) == sorted(_CASTS | _REFUSED)
    for name, dtype in _CASTS.items():
        assert getattr(ref.ones(1), name)().dtype == getattr(
            ref, dtype.removesuffix("_")
        )
