"""``load_state_dict(assign=True)`` checks shapes and runs each module's loader.

The assign path called the default loader directly: it skipped the shape
check, so ``Linear(2, 2)`` took a ``(3, 3)`` weight, and it skipped every
module's own ``_load_from_state_dict`` — a lazy layer was never sized, a
version-1 BatchNorm checkpoint missed ``num_batches_tracked`` (LCD-499).
The reference sends an assign load through the same ``_load_from_state_dict``
as a copy, telling it by ``local_metadata["assign_to_params_buffers"]``, and
checks the shape on both.
"""

import re
from collections import OrderedDict

import pytest

import lucid
import lucid.nn as nn
from lucid.quantization.observer import PerChannelMinMaxObserver


def _shapes_as_tuples(message):
    # The reference prints a shape as ``<lib>.Size([3, 3])``, Lucid as a tuple.
    def as_tuple(m):
        return str(tuple(int(d) for d in m.group(1).split(",") if d.strip()))

    return re.sub(r"\w+\.Size\(\[([^\]]*)\]\)", as_tuple, message)


def _bad_linear(lib):
    return lib.nn.Linear(2, 2), {"weight": lib.zeros(3, 3), "bias": lib.zeros(2)}


def _bad_batchnorm(lib):
    module = lib.nn.BatchNorm1d(2)
    state = dict(module.state_dict())
    state["running_mean"] = lib.zeros(3)
    return module, state


_BAD = {"parameter": _bad_linear, "buffer": _bad_batchnorm}


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "not strict"])
@pytest.mark.parametrize("assign", [True, False], ids=["assign", "copy"])
@pytest.mark.parametrize("case", sorted(_BAD))
def test_a_shape_mismatch_is_refused(case, assign, strict, device):
    module, state = _BAD[case](lucid)
    module.to(device)
    before = {k: (v, tuple(v.shape)) for k, v in module.state_dict(keep_vars=True).items()}
    with pytest.raises(RuntimeError, match="size mismatch for"):
        module.load_state_dict(state, strict=strict, assign=assign)
    after = module.state_dict(keep_vars=True)
    (name,) = [k for k, v in state.items() if tuple(v.shape) != before[k][1]]
    assert after[name] is before[name][0] and tuple(after[name].shape) == before[name][1]


@pytest.mark.parity
@pytest.mark.parametrize("assign", [True, False], ids=["assign", "copy"])
@pytest.mark.parametrize("case", sorted(_BAD))
def test_the_message_is_the_reference_message(ref, case, assign):
    def message(lib):
        module, state = _BAD[case](lib)
        with pytest.raises(RuntimeError) as info:
            module.load_state_dict(state, assign=assign)
        return _shapes_as_tuples(str(info.value))

    assert message(lucid) == message(ref)


def test_an_assign_load_still_takes_the_checkpoint_dtype(device):
    module = nn.Linear(2, 2).to(device)
    state = {
        "weight": lucid.zeros(2, 2, dtype=lucid.float16, device=device),
        "bias": lucid.zeros(2, dtype=lucid.float16, device=device),
    }
    module.load_state_dict(state, assign=True)
    assert module.weight.dtype == lucid.float16
    assert type(module.weight) is nn.Parameter and module.weight.requires_grad


# Each module below has its own ``_load_from_state_dict``; an assign load
# that bypassed it failed or left the module half-built.
def _lazy_linear():
    module = nn.LazyLinear(3)
    module.load_state_dict({"weight": lucid.ones(3, 4), "bias": lucid.ones(3)}, assign=True)
    assert module.in_features == 4 and tuple(module.weight.shape) == (3, 4)
    assert module(lucid.ones(1, 4)).shape == (1, 3)


def _lazy_conv():
    module = nn.LazyConv2d(3, 1)
    state = {"weight": lucid.ones(3, 2, 1, 1), "bias": lucid.ones(3)}
    module.load_state_dict(state, assign=True)
    out = module(lucid.ones(1, 2, 4, 4))
    assert module.in_channels == 2 and out.shape == (1, 3, 4, 4)


def _batchnorm_version_1():
    state = OrderedDict(
        (k, v) for k, v in nn.BatchNorm1d(2).state_dict().items() if k != "num_batches_tracked"
    )
    state._metadata = {"": {"version": 1}}
    module = nn.BatchNorm1d(2)
    module.load_state_dict(state, assign=True)
    assert int(module.num_batches_tracked.item()) == 0


def _calibrated_observer():
    calibrated = PerChannelMinMaxObserver(ch_axis=0)
    calibrated(lucid.randn(3, 4))
    fresh = PerChannelMinMaxObserver(ch_axis=0)
    fresh.load_state_dict(calibrated.state_dict(), assign=True)
    assert tuple(fresh.min_val.shape) == (3,)


def _attention_reference_layout():
    source = nn.MultiheadAttention(4, 2)
    state = OrderedDict()
    for k, v in source.state_dict().items():
        state[k.replace("out_proj_", "out_proj.")] = v
    module = nn.MultiheadAttention(4, 2)
    module.load_state_dict(state, assign=True)
    assert module.out_proj_weight.tolist() == source.out_proj_weight.tolist()


_OWN_LOADERS = {
    "lazy linear": _lazy_linear,
    "lazy conv": _lazy_conv,
    "batchnorm version 1": _batchnorm_version_1,
    "calibrated observer": _calibrated_observer,
    "attention in the reference layout": _attention_reference_layout,
}


@pytest.mark.parametrize("case", sorted(_OWN_LOADERS))
def test_a_modules_own_loader_runs_on_an_assign_load(case):
    _OWN_LOADERS[case]()


def test_the_checkpoint_metadata_is_left_alone():
    module = nn.Linear(2, 2)
    state = module.state_dict()
    module.load_state_dict(state, assign=True)
    assert "assign_to_params_buffers" not in state._metadata[""]
