"""``load_state_dict(assign=True)`` carries the module's flag onto Parameters only.

A Parameter put in place keeps the ``requires_grad`` of the one it replaces
— frozen stays frozen — but a buffer is the loaded tensor itself, its own
flag included.  Lucid gave the buffer the old buffer's flag too, so a
checkpoint tensor that required grad lost it on the way in (LCD-486).
"""

import pytest

import lucid
import lucid.nn as nn


def _state(module, lib, *, buffers_grad):
    """``module``'s state, every float buffer requiring grad if ``buffers_grad``."""
    out = {}
    names = dict(module.named_buffers())
    for key, value in module.state_dict().items():
        v = value.clone()
        if buffers_grad and key in names and v.is_floating_point():
            v.requires_grad_(True)
        out[key] = v
    return out


def _flags(module):
    return {
        name: (getattr(module, name).requires_grad, type(getattr(module, name)).__name__)
        for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    }


@pytest.mark.parametrize("buffers_grad", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_parameters_keep_the_module_flag_and_buffers_their_own(buffers_grad, frozen):
    target = nn.BatchNorm1d(3)
    target.weight.requires_grad_(not frozen)
    target.load_state_dict(_state(nn.BatchNorm1d(3), lucid, buffers_grad=buffers_grad), assign=True)
    flags = _flags(target)
    assert flags["weight"] == (not frozen, "Parameter")
    assert flags["bias"] == (True, "Parameter")
    assert flags["running_mean"][0] is buffers_grad
    assert flags["running_var"][0] is buffers_grad
    assert flags["num_batches_tracked"][0] is False


@pytest.mark.parity
@pytest.mark.parametrize("buffers_grad", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_the_flags_match_the_reference(ref, buffers_grad, frozen):
    def run(lib):
        target = lib.nn.BatchNorm1d(3)
        target.weight.requires_grad_(not frozen)
        source = _state(lib.nn.BatchNorm1d(3), lib, buffers_grad=buffers_grad)
        target.load_state_dict(source, assign=True)
        return _flags(target)

    assert run(lucid) == run(ref)
