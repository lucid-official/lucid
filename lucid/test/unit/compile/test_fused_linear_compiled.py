"""``nn.FusedLinear`` stays on the graph under ``lucid.compile`` (LCD-450).

The inference kernels ``_fused_linear_relu`` / ``_fused_linear_gelu`` are
recorded as one op each (LCD-345), and had no MPSGraph emitter, so a compiled
``FusedLinear`` ran eagerly with a fallback notice. On Metal they did not run
at all before then: only the CPU backend had the kernel, for float32.

The emitter spells the eager function — the GEMM with the weight transposed,
the bias, then ``relu`` or the tanh-form GELU — and the compiled answer has
to agree with eager to a couple of units in the last place of the dtype.
"""

import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.test.unit.compile._host_reads import FUSED_KERNELS, Apply, inputs


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

#: Machine epsilon per dtype. The bound is a few of them at the output's
#: scale: MPSGraph's and MLX's GEMMs accumulate in their own order.
_EPS = {lucid.float32: 2.0**-23, lucid.float16: 2.0**-10, lucid.bfloat16: 2.0**-7}


@pytest.mark.parametrize("dtype", list(_EPS), ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_a_compiled_fused_linear_is_one_executable(
    activation: str, dtype: lucid.dtype
) -> None:
    lucid.manual_seed(0)
    model = nn.FusedLinear(8, 16, activation=activation).to("metal").to(dtype).eval()
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model)
    calls = [lucid.randn(2, 3, 8).to("metal").to(dtype) for _ in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(x) for x in calls]
            want = [model(x) for x in calls]

    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    eps = _EPS[dtype]
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape == (2, 3, 16)
        assert g.dtype == w.dtype
        scale = max(1.0, float(w.float().abs().max().item()))
        assert float((g.float() - w.float()).abs().max().item()) <= 4 * eps * scale


@pytest.mark.parametrize("name", sorted(FUSED_KERNELS))
def test_a_fused_kernel_follows_the_input_compiled(name: str) -> None:
    """LCD-345's fused-kernel cases, now on Metal: one executable whose
    second answer is eager's, not the first call's."""
    fn = FUSED_KERNELS[name]
    first, second = inputs()
    reference = Apply(fn).eval()
    want = [reference(t).to("cpu") for t in (first, second)]
    assert float((want[0] - want[1]).abs().max().item()) > 1e-3

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(Apply(fn).eval().to("metal"))
    # Under grad mode a trainable model's call runs eagerly (LCD-582).
    with warnings.catch_warnings(), lucid.no_grad():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        got = [compiled(t.to("metal")).to("cpu") for t in (first, second)]
    assert not compiled.cache_info()["eager_only"]
    for g, w in zip(got, want, strict=True):
        assert float((g - w).abs().max().item()) <= 1e-4 * float(w.abs().max().item())


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_the_trace_holds_the_fused_op_alone(activation: str) -> None:
    """The fused binding records one op; nothing of the composition leaks."""
    model = nn.FusedLinear(8, 4, activation=activation).to("metal").eval()
    x = lucid.randn(5, 8).to("metal")
    with lucid.no_grad(), lucid.compile._tracing(watch=((x,), {})) as tracer:
        model(x)
    names = [op.name for op in tracer.graph.ops]
    assert names == [f"_fused_linear_{activation}"], names
