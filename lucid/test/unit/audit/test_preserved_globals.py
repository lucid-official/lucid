"""``_probe.preserved_globals`` puts back every process-wide switch a probed symbol can flip.

The audit and the bool-refusal sweep call every public symbol, setters included.  A switch
left flipped re-tunes every test that runs later in the same process: determinism left on
made ``dropout`` refuse in the dtype-parity tests that followed the sweep (LCD-1020).
"""

import lucid
from lucid.test.audit import _probe


def test_determinism_is_restored() -> None:
    before = lucid.are_deterministic_algorithms_enabled()
    with _probe.preserved_globals():
        lucid.use_deterministic_algorithms(not before)
    assert lucid.are_deterministic_algorithms_enabled() is before


def test_grad_mode_dtype_and_device_are_restored() -> None:
    dtype, device, grad = (
        lucid.get_default_dtype(),
        lucid.get_default_device(),
        lucid.is_grad_enabled(),
    )
    with _probe.preserved_globals():
        lucid.set_default_dtype(lucid.float64 if dtype is not lucid.float64 else lucid.float32)
        lucid.set_grad_enabled(not grad)
    assert lucid.get_default_dtype() is dtype
    assert lucid.get_default_device() == device
    assert lucid.is_grad_enabled() is grad
