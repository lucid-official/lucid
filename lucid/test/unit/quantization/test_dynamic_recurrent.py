"""``quantize_dynamic`` covers every module type the reference's dynamic mode does.

Only ``Linear`` and ``LSTM`` used to be mapped, so a ``GRU`` or an
``LSTMCell`` / ``GRUCell`` / ``RNNCell`` came back as the float module with
float32 weights and no warning (LCD-190).  The matrix here is every mapped
type × CPU / Metal: the quantized module is the dynamic form, holds int8
weights and no float weight in its ``state_dict``, and its output stays close
to the float module's.  A type the caller names that has no dynamic form
(``Conv2d``, the multi-step ``RNN``) stays float with a warning.
"""

import contextlib
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized.dynamic as nnqd
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    """Pin the dequant path so the Linear swap is the dynamic Linear."""
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


def _sequence(lib: Any) -> Any:
    return lib.randn(5, 2, 4)


def _step(lib: Any) -> Any:
    return lib.randn(3, 4)


# name → (float module factory taking the library, input factory, quantized type)
_CASES: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any], type]] = {
    "Linear": (lambda lib: lib.nn.Linear(4, 6), _step, nnqd.Linear),
    "LSTM": (lambda lib: lib.nn.LSTM(4, 6, num_layers=2), _sequence, nnqd.LSTM),
    "GRU": (lambda lib: lib.nn.GRU(4, 6), _sequence, nnqd.GRU),
    "GRU-2layer-bidir": (
        lambda lib: lib.nn.GRU(4, 6, num_layers=2, bidirectional=True),
        _sequence,
        nnqd.GRU,
    ),
    "LSTMCell": (lambda lib: lib.nn.LSTMCell(4, 6), _step, nnqd.LSTMCell),
    "GRUCell": (lambda lib: lib.nn.GRUCell(4, 6), _step, nnqd.GRUCell),
    "RNNCell-tanh": (lambda lib: lib.nn.RNNCell(4, 6), _step, nnqd.RNNCell),
    "RNNCell-relu": (lambda lib: lib.nn.RNNCell(4, 6, nonlinearity="relu"), _step, nnqd.RNNCell),
}


def _first(out: Any) -> Any:
    """The output tensor of a module: a cell's ``h``, a sequence layer's output."""
    return out[0] if isinstance(out, tuple) else out


def _host(t: Any) -> np.ndarray:
    if isinstance(t, lucid.Tensor):
        return t.detach().to("cpu").numpy().astype(np.float64)
    return t.detach().double().numpy()


def _rel_err(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).mean() / (np.abs(want).mean() + 1e-9))


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("case", list(_CASES))
def test_each_type_is_quantized_and_stays_close(case: str, device: str) -> None:
    make, inp, qtype = _CASES[case]
    lucid.manual_seed(0)
    model = nn.Sequential(make(lucid)).to(device)
    model.eval()
    with _reference_engine(), warnings.catch_warnings():
        warnings.simplefilter("error")
        qmodel = Q.quantize_dynamic(model)
    assert type(qmodel[0]) is qtype
    state = qmodel.state_dict()
    codes = [k for k in state if k.endswith("_int8")]
    assert codes
    assert all(state[k].dtype is lucid.int8 for k in codes)
    # No float weight leaks from the shell: every weight is stored as int8 codes
    # with their quantization parameters.
    qparts = ("_int8", "_scale", "_zero_point")
    assert not [k for k in state if ".weight" in k and not k.endswith(qparts)]
    x = inp(lucid).to(device)
    want = _host(_first(model(x)))
    got = _host(_first(qmodel(x)))
    assert _rel_err(got, want) < 0.03


@pytest.mark.parametrize("case", ["GRU-2layer-bidir", "LSTMCell", "GRUCell"])
def test_quantized_module_follows_a_device_move(case: str) -> None:
    make, inp, _ = _CASES[case]
    lucid.manual_seed(1)
    model = make(lucid)
    model.eval()
    qmodel = Q.quantize_dynamic(model).to("metal")
    x = inp(lucid)
    want = _host(_first(model(x)))
    got = _host(_first(qmodel(x.to("metal"))))
    assert _rel_err(got, want) < 0.03


def test_default_spec_quantizes_every_recurrent_type() -> None:
    lucid.manual_seed(2)
    model = nn.Sequential(nn.GRU(4, 4), nn.LSTMCell(4, 4), nn.GRUCell(4, 4), nn.RNNCell(4, 4))
    qmodel = Q.quantize_dynamic(model)
    assert [type(m) for m in qmodel] == [nnqd.GRU, nnqd.LSTMCell, nnqd.GRUCell, nnqd.RNNCell]


def test_spec_filters_recurrent_types() -> None:
    model = nn.Sequential(nn.GRU(4, 4), nn.GRUCell(4, 4))
    qmodel = Q.quantize_dynamic(model, qconfig_spec={nn.GRUCell})
    assert type(qmodel[0]) is nn.GRU
    assert type(qmodel[1]) is nnqd.GRUCell


@pytest.mark.parametrize("named", [nn.Conv2d, nn.RNN])
def test_named_type_without_a_dynamic_form_warns(named: type) -> None:
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.RNN(2, 2), nn.Linear(2, 2))
    with pytest.warns(UserWarning, match=f"{named.__name__} has no dynamically"):
        qmodel = Q.quantize_dynamic(model, qconfig_spec={named})
    assert [type(m) for m in qmodel] == [nn.Conv2d, nn.RNN, nn.Linear]


@pytest.mark.parametrize("case", ["LSTM", "GRU-2layer-bidir", "LSTMCell", "RNNCell-relu"])
def test_from_float_leaves_the_float_module_alone(case: str) -> None:
    make, inp, qtype = _CASES[case]
    lucid.manual_seed(3)
    model = make(lucid)
    before = {k: v.numpy().copy() for k, v in model.state_dict().items()}
    qmod = qtype.from_float(model)
    qmod(inp(lucid))
    after = model.state_dict()
    assert all(np.array_equal(before[k], after[k].numpy()) for k in before)
    assert all(p.requires_grad for p in model.parameters())


def test_from_float_rejects_another_type() -> None:
    with pytest.raises(TypeError, match="expects an nn.GRUCell"):
        nnqd.GRUCell.from_float(nn.LSTMCell(2, 2))


# ── against the reference ────────────────────────────────────────────────────


@contextlib.contextmanager
def _ref_qengine(ref: Any) -> Iterator[None]:
    prev = ref.backends.quantized.engine
    ref.backends.quantized.engine = "qnnpack"
    try:
        yield
    finally:
        # The unset engine ("none") is not one the setter accepts back.
        if prev in ref.backends.quantized.supported_engines:
            ref.backends.quantized.engine = prev


@pytest.mark.parity
def test_default_spec_matches_the_reference(ref: Any) -> None:
    def build(lib: Any) -> Any:
        return lib.nn.Sequential(
            lib.nn.Linear(4, 4),
            lib.nn.LSTM(4, 4),
            lib.nn.GRU(4, 4),
            lib.nn.LSTMCell(4, 4),
            lib.nn.GRUCell(4, 4),
            lib.nn.RNNCell(4, 4),
            lib.nn.RNN(4, 4),
            lib.nn.Conv2d(1, 1, 1),
        )

    with _ref_qengine(ref), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = [
            type(m).__module__.startswith(ref.__name__ + ".ao.")
            for m in ref.ao.quantization.quantize_dynamic(build(ref))
        ]
    with _reference_engine():
        got = [
            type(m).__module__.startswith("lucid.nn.quantized")
            for m in Q.quantize_dynamic(build(lucid))
        ]
    assert got == want


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CASES))
def test_output_matches_the_reference(case: str, ref: Any) -> None:
    make, inp, _ = _CASES[case]
    ref.manual_seed(0)
    float_ref = make(ref).eval()
    model = make(lucid)
    model.load_state_dict({k: lucid.tensor(v.numpy()) for k, v in float_ref.state_dict().items()})
    model.eval()
    with _ref_qengine(ref), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qref = ref.ao.quantization.quantize_dynamic(ref.nn.Sequential(float_ref))
    with _reference_engine():
        qmodel = Q.quantize_dynamic(model)
    x = inp(ref)
    want = _host(_first(qref(x)))
    got = _host(_first(qmodel(lucid.tensor(x.numpy()))))
    # Both are int8-weight approximations of the same float module; the
    # reference also quantizes activations, so they agree to its error.
    assert _rel_err(got, want) < 0.03
