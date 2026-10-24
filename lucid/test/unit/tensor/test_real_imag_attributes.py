"""``Tensor.real`` / ``Tensor.imag`` and ``lucid.real`` of a real tensor (LCD-547).

The attributes did not exist — only the functions — and ``lucid.real`` of a
real tensor raised ``NotImplementedError`` where the reference hands the
tensor back.  The attributes are the functions' views of the complex
tensor's own storage, written through on Metal as on the CPU, and can be
assigned to.
"""

from types import ModuleType

import pytest

import lucid


def test_real_of_a_real_tensor_is_the_tensor(device: str) -> None:
    x = lucid.tensor([1.0, -2.0], device=device)
    assert lucid.real(x) is x
    assert x.real is x
    i = lucid.tensor([1, 2], device=device)
    assert lucid.real(i) is i


def test_imag_of_a_real_tensor_is_refused(device: str) -> None:
    x = lucid.tensor([1.0, -2.0], device=device)
    with pytest.raises(RuntimeError):
        x.imag
    with pytest.raises(RuntimeError):
        lucid.imag(x)


@pytest.mark.parametrize(
    "dtype,part", [("complex64", "float32"), ("complex128", "float64")]
)
def test_the_parts_are_views_of_the_complex_storage(
    device: str, dtype: str, part: str
) -> None:
    if device == "metal" and dtype == "complex128":
        pytest.skip("Metal holds no complex128")
    z = lucid.tensor([1 + 2j, 3 - 4j], dtype=getattr(lucid, dtype), device=device)
    re, im = z.real, z.imag
    assert re.dtype == im.dtype == getattr(lucid, part)
    assert (re.tolist(), im.tolist()) == ([1.0, 3.0], [2.0, -4.0])
    re.fill_(9.0)
    im.mul_(2.0)
    assert z.tolist() == [9 + 4j, 9 - 8j]


def test_assigning_a_part_writes_it_in_place(device: str) -> None:
    z = lucid.tensor([1 + 2j, 3 - 4j], device=device)
    view = z[1:]
    z.real = lucid.tensor([5.0, 6.0], device=device)
    z.imag = 0.5
    assert z.tolist() == [5 + 0.5j, 6 + 0.5j]
    assert view.tolist() == [6 + 0.5j]
    x = lucid.zeros(2, device=device)
    x.real = 3.0
    assert x.tolist() == [3.0, 3.0]
    with pytest.raises(RuntimeError):
        x.imag = 1.0


def test_the_real_part_carries_the_gradient() -> None:
    z = lucid.tensor([1 + 2j, 3 - 4j], requires_grad=True)
    (z.real * lucid.tensor([2.0, 3.0])).sum().backward()
    assert z.grad is not None
    assert z.grad.tolist() == [2 + 0j, 3 + 0j]


@pytest.mark.parity
def test_the_attributes_match_the_reference(ref: ModuleType) -> None:
    data = [1 + 2j, 3 - 4j]
    ours = lucid.tensor(data)
    theirs = ref.tensor(data)
    assert ours.real.tolist() == theirs.real.tolist()
    assert ours.imag.tolist() == theirs.imag.tolist()
    ours.real, theirs.real = 7.0, 7.0
    assert ours.tolist() == theirs.tolist()
    real = ref.tensor([1.0, 2.0])
    assert ref.real(real) is real
    with pytest.raises(RuntimeError):
        real.imag
