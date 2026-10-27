"""``_amp_op`` gives a composite its row at its entry, for the whole call (LCD-614).

The losses applied their FP32 row only through their reduction, so
``reduction="none"`` kept a half input half.  Each now takes the row at its
entry with ``_amp_op``: every tensor argument is cast by the row, a
``weight`` as well as the input (the reference's autocast casts every
floating tensor an op is handed), and the body runs with autocast off.
These pin the decorator's own contract.
"""

import inspect

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops._autocast import _amp_op
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_SETTINGS = [
    pytest.param("cpu", "bfloat16", id="cpu-bfloat16"),
    pytest.param(
        "metal",
        "float16",
        id="metal-float16",
        marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
    ),
]

#: What the decorated body saw on its last call.
_seen: dict[str, object] = {}


@_amp_op("l1_loss")  # FP32 on both devices
def _composite(
    input: lucid.Tensor,
    target: lucid.Tensor,
    scale: float = 2.0,
    *,
    weight: lucid.Tensor | None = None,
    index: lucid.Tensor | None = None,
) -> lucid.Tensor:
    """Record what arrived, then compute with it."""
    device = input._impl.device
    _seen.update(
        input=input.dtype,
        target=target.dtype,
        weight=None if weight is None else weight.dtype,
        index=None if index is None else index.dtype,
        scale=scale,
        autocast=_C_engine.amp_is_active(device),
    )
    out = (input - target) * scale
    return out if weight is None else out * weight


@_amp_op("bce_loss")  # KEEP on Metal, FP32 on the CPU
def _keeps_on_metal(input: lucid.Tensor) -> lucid.Tensor:
    _seen.update(input=input.dtype, autocast=_C_engine.amp_is_active(input._impl.device))
    return input * 2.0


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_every_floating_tensor_argument_is_cast_by_the_row(device: str, half: str) -> None:
    dt = getattr(lucid, half)
    x = lucid.randn(3, device=device).to(dt)
    with autocast(device_type=device, dtype=dt):
        out = _composite(x, x, weight=lucid.ones(3, device=device).to(dt))
    assert out.dtype == lucid.float32
    assert _seen["input"] == _seen["target"] == _seen["weight"] == lucid.float32


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_a_tensor_passed_by_position_or_keyword_is_cast_alike(device: str, half: str) -> None:
    dt = getattr(lucid, half)
    x = lucid.randn(3, device=device).to(dt)
    with autocast(device_type=device, dtype=dt):
        _composite(target=x, input=x)
    assert _seen["input"] == _seen["target"] == lucid.float32


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_what_is_not_a_floating_tensor_passes_through(device: str, half: str) -> None:
    dt = getattr(lucid, half)
    x = lucid.randn(3, device=device).to(dt)
    with autocast(device_type=device, dtype=dt):
        _composite(x, x, 3.0, index=lucid.tensor([0, 2], device=device))
    assert _seen["index"] == lucid.int64
    assert _seen["scale"] == 3.0
    assert _seen["weight"] is None


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_the_body_runs_with_autocast_off_and_it_is_back_on_after(device: str, half: str) -> None:
    dt = getattr(lucid, half)
    x = lucid.randn(3, device=device).to(dt)
    with autocast(device_type=device, dtype=dt):
        _composite(x, x)
        assert _C_engine.amp_is_active(x._impl.device)
    assert _seen["autocast"] is False


@_amp_op("l1_loss")
def _refuses(input: lucid.Tensor) -> lucid.Tensor:
    raise ValueError(f"refused a {input.dtype} input")


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_autocast_comes_back_when_the_body_raises(device: str, half: str) -> None:
    dt = getattr(lucid, half)
    x = lucid.randn(3, device=device).to(dt)
    with autocast(device_type=device, dtype=dt):
        with pytest.raises(ValueError, match="float32"):
            _refuses(x)
        assert _C_engine.amp_is_active(x._impl.device)


def test_the_column_is_the_device_s() -> None:
    """``bce_loss`` keeps a float16 input on Metal and widens it on the CPU."""
    x = lucid.randn(3).to(lucid.bfloat16)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        assert _keeps_on_metal(x).dtype == lucid.float32
    if metal_available():
        y = lucid.randn(3, device="metal").to(lucid.float16)
        with autocast(device_type="metal", dtype=lucid.float16):
            assert _keeps_on_metal(y).dtype == lucid.float16
        assert _seen["input"] == lucid.float16 and _seen["autocast"] is False


def test_outside_autocast_the_arguments_arrive_as_they_were_passed() -> None:
    x = lucid.randn(3).to(lucid.bfloat16)
    _composite(x, x)
    assert _seen["input"] == lucid.bfloat16


def test_a_float32_argument_arrives_as_the_same_tensor() -> None:
    """The row leaves a float32 argument alone, and it arrives unwrapped."""
    got: list[lucid.Tensor] = []

    @_amp_op("l1_loss")
    def keep(input: lucid.Tensor) -> lucid.Tensor:
        got.append(input)
        return input

    x = lucid.randn(3)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        keep(x)
    assert got[0] is x


def test_the_signature_name_and_docstring_are_the_function_s() -> None:
    def plain(
        input: lucid.Tensor,
        target: lucid.Tensor,
        scale: float = 2.0,
        *,
        weight: lucid.Tensor | None = None,
        index: lucid.Tensor | None = None,
    ) -> lucid.Tensor:
        raise NotImplementedError

    assert inspect.signature(_composite) == inspect.signature(plain)
    assert _composite.__name__ == "_composite"
    assert _composite.__module__ == __name__
    assert _composite.__doc__ == "Record what arrived, then compute with it."


#: What a caller binds to, spelled out: ``inspect.signature`` follows
#: ``__wrapped__``, so comparing with the undecorated def could not fail.
_LOSS_PARAMS = {
    "binary_cross_entropy_with_logits": [
        "input",
        "target",
        "weight=",
        "size_average=",
        "reduce=",
        "reduction=",
        "pos_weight=",
    ],
    "cross_entropy": [
        "input",
        "target",
        "weight=",
        "size_average=",
        "ignore_index=",
        "reduce=",
        "reduction=",
        "label_smoothing=",
    ],
    "kl_div": [
        "input",
        "target",
        "size_average=",
        "reduce=",
        "reduction=",
        "log_target=",
    ],
    "ctc_loss": [
        "log_probs",
        "targets",
        "input_lengths",
        "target_lengths",
        "blank=",
        "reduction=",
        "zero_infinity=",
    ],
}


@pytest.mark.parametrize("name", sorted(_LOSS_PARAMS))
def test_a_decorated_loss_keeps_its_signature(name: str) -> None:
    """What a caller binds to, and what the stubs and docs read, is the def's."""
    fn = getattr(lucid.nn.functional, name)
    params = [
        p.name + ("" if p.default is p.empty else "=")
        for p in inspect.signature(fn).parameters.values()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    assert len(params) == len(inspect.signature(fn).parameters)
    assert params == _LOSS_PARAMS[name]
    assert fn.__module__ == "lucid.nn.functional.loss"
    assert fn.__name__ == name and fn.__doc__ == fn.__wrapped__.__doc__


@pytest.mark.parametrize(("device", "half"), _SETTINGS)
def test_a_call_that_does_not_bind_raises_the_function_s_own_error(device: str, half: str) -> None:
    """``signature.bind`` would refuse first, with its own wording; the
    error under autocast is the one the call raises outside it."""
    dt = getattr(lucid, half)
    x = lucid.randn(3, 4, device=device).to(dt)
    F = lucid.nn.functional
    with pytest.raises(TypeError) as outside:
        F.cross_entropy(x)  # type: ignore[call-arg]
    with autocast(device_type=device, dtype=dt):
        with pytest.raises(TypeError) as inside:
            F.cross_entropy(x)  # type: ignore[call-arg]
        with pytest.raises(TypeError) as unknown:
            F.nll_loss(x, x, bogus=1)  # type: ignore[call-arg]
    assert str(inside.value) == str(outside.value)
    assert "cross_entropy()" in str(inside.value)
    assert "bogus" in str(unknown.value)
