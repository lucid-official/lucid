"""Export refuses a model that reads a traced value on the host (LCD-337).

The refusal used to fire only when an output was missing from the trace.
``x * 0 + lucid.full_like(x, x.sum().item())`` traces whole — the read
value enters as a constant — so it exported with the example's sum baked
in and answered every later input with it. The tracer marks any host read
of a traced value, and of an argument before an op has consumed it, and
export refuses the marked trace whatever its outputs look like.

The same reads, as the compile path meets them, are in
``lucid/test/unit/compile/test_host_reads.py``; there they run eagerly.
"""

import pytest

import lucid
import lucid.coreml as cml
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._host_reads import (
    CONSTANT_READS,
    FUSED_KERNELS,
    HOST_READS,
    STATE_READS,
    Apply,
    inputs,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


@pytest.mark.parametrize("name", sorted(set(HOST_READS) - STATE_READS))
def test_a_host_read_is_refused(name: str, tmp_path: object) -> None:
    first, _second = inputs()
    with pytest.raises(ValueError, match="read on the host"):
        cml.export(Apply(HOST_READS[name]).eval(), first, f"{tmp_path}/{name}.mlpackage")


_EXPORTABLE = {**CONSTANT_READS, **{n: HOST_READS[n] for n in STATE_READS}}


@pytest.mark.parametrize("name", sorted(_EXPORTABLE))
def test_a_read_of_a_constant_exports(name: str, tmp_path: object) -> None:
    """A tensor built from Python numbers, a validation-only read, or the
    model's weights — which a package holds as constants — all export."""
    first, second = inputs()
    model = Apply(_EXPORTABLE[name]).eval()
    exported = cml.export(model, first, f"{tmp_path}/{name}.mlpackage")
    got = exported.predict(second)
    assert isinstance(got, lucid.Tensor)
    assert float((got - model(second)).abs().max().item()) <= 1e-5


@pytest.mark.parametrize("name", sorted(FUSED_KERNELS))
def test_a_fused_kernel_exports_and_follows_the_input(name: str, tmp_path: object) -> None:
    """``F.fused_linear_relu(x, w, b) * 2 + x.sum()`` exported with the
    example's kernel output baked in, off from eager by about 4 on another
    input (LCD-345). The kernel is recorded, and translates to MIL's linear
    and activation (LCD-450), so the package answers a new input as eager."""
    first, second = inputs()
    model = Apply(FUSED_KERNELS[name]).eval()
    exported = cml.export(model, first, f"{tmp_path}/{name}.mlpackage")
    want = model(second)
    got = exported.predict(second)
    assert isinstance(got, lucid.Tensor)
    assert float((got - want).abs().max().item()) <= 1e-4 * float(want.abs().max().item())
