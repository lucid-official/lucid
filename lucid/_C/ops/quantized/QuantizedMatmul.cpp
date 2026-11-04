// lucid/_C/ops/quantized/QuantizedMatmul.cpp
//
// GPU-stream int4/int8 GEMM primitives over MLX group-wise affine kernels.
// Pattern B (see ops/fft, ops/linalg GPU paths): call mlx::core directly,
// guarded to Device::GPU, wrap the result back, and wire ``QuantizedMatmulBackward``
// for ``dX`` so QLoRA adapters can backpropagate through frozen quantized layers.

#include "QuantizedMatmul.h"

#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Matmul.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Transpose.h"

namespace lucid {
namespace {

// Extract the backing ``mlx::core::array`` from a GPU tensor (guarded).
::mlx::core::array gpu_array(const TensorImplPtr& t, const char* who) {
    if (!t)
        ErrorBuilder(who).fail("null input tensor");
    if (t->device() != Device::GPU)
        ErrorBuilder(who).fail("quantized ops require GPU (Metal) tensors");
    return *std::get<GpuStorage>(t->storage()).arr;
}

// Wrap an MLX result into a fresh GPU TensorImpl tagged with ``dt``.
TensorImplPtr wrap(::mlx::core::array&& a, Dtype dt) {
    Shape sh = gpu::mlx_shape_to_lucid(a.shape());
    Storage st{gpu::wrap_mlx_array(std::move(a), dt)};
    return ::lucid::helpers::fresh(std::move(st), sh, dt, Device::GPU);
}

// The packed weight is uint32 but Lucid tags it I32 (no uint32 dtype).  After a
// serialize / reload round-trip it comes back as a genuine int32 array, which
// MLX's quantized kernels reject.  A zero-cost bitcast to uint32 makes the op
// robust whether the tensor is still uint32 (in-memory) or int32 (reloaded).
::mlx::core::array as_packed_u32(const ::mlx::core::array& w) {
    return ::mlx::core::view(w, ::mlx::core::uint32);
}

}  // namespace

const OpSchema QuantizedMatmulBackward::schema_v1{"quantized_matmul", 1, AmpPolicy::KeepInput,
                                                  true};

std::vector<Storage> QuantizedMatmulBackward::apply(Storage grad_out) {
    const ::mlx::core::array& ga = *std::get<GpuStorage>(grad_out).arr;
    ::mlx::core::array wa = as_packed_u32(*std::get<GpuStorage>(saved_w_).arr);
    const ::mlx::core::array& sa = *std::get<GpuStorage>(saved_scales_).arr;
    std::optional<::mlx::core::array> ba =
        saved_biases_.has_value()
            ? std::optional<::mlx::core::array>(*std::get<GpuStorage>(*saved_biases_).arr)
            : std::nullopt;
    ::mlx::core::array w_deq = ::mlx::core::dequantize(wa, sa, ba, group_size_, bits_);
    if (!transpose_)
        w_deq = ::mlx::core::swapaxes(w_deq, -1, -2);
    if (w_deq.dtype() != ga.dtype())
        w_deq = ::mlx::core::astype(w_deq, ga.dtype());
    ::mlx::core::array dx = ::mlx::core::matmul(ga, w_deq);
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)}};
}

std::vector<TensorImplPtr> QuantizedMatmulBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    auto w = std::make_shared<TensorImpl>(saved_w_, w_shape_, Dtype::I32, Device::GPU, false);
    auto scales =
        std::make_shared<TensorImpl>(saved_scales_, scales_shape_, Dtype::F32, Device::GPU, false);
    TensorImplPtr biases = nullptr;
    if (saved_biases_.has_value()) {
        biases = std::make_shared<TensorImpl>(*saved_biases_, biases_shape_, Dtype::F32,
                                              Device::GPU, false);
    }
    TensorImplPtr w_deq = dequantize_op(w, scales, biases, group_size_, bits_);
    if (!transpose_)
        w_deq = mT_op(w_deq);
    if (w_deq->dtype() != grad_out->dtype())
        w_deq = astype_op(w_deq, grad_out->dtype());
    return {matmul_op(grad_out, w_deq)};
}

LUCID_REGISTER_OP(QuantizedMatmulBackward)

std::vector<TensorImplPtr> quantize_op(const TensorImplPtr& w, int group_size, int bits) {
    ::mlx::core::array wa = gpu_array(w, "quantize");
    if (wa.dtype() != ::mlx::core::float32)
        wa = ::mlx::core::astype(wa, ::mlx::core::float32);
    std::vector<::mlx::core::array> res = ::mlx::core::quantize(wa, group_size, bits);
    return {
        wrap(std::move(res[0]), Dtype::I32),  // packed uint32 → I32-tagged opaque
        wrap(std::move(res[1]), Dtype::F32),  // scales
        wrap(std::move(res[2]), Dtype::F32),  // biases
    };
}

TensorImplPtr dequantize_op(const TensorImplPtr& w,
                            const TensorImplPtr& scales,
                            const TensorImplPtr& biases,
                            int group_size,
                            int bits) {
    ::mlx::core::array wa = as_packed_u32(gpu_array(w, "dequantize"));
    ::mlx::core::array sa = gpu_array(scales, "dequantize");
    if (sa.dtype() != ::mlx::core::float32)
        sa = ::mlx::core::astype(sa, ::mlx::core::float32);
    std::optional<::mlx::core::array> ba = std::nullopt;
    if (biases) {
        ::mlx::core::array b = gpu_array(biases, "dequantize");
        if (b.dtype() != ::mlx::core::float32)
            b = ::mlx::core::astype(b, ::mlx::core::float32);
        ba = std::move(b);
    }
    ::mlx::core::array out = ::mlx::core::dequantize(wa, sa, ba, group_size, bits);
    return wrap(std::move(out), Dtype::F32);
}

TensorImplPtr quantized_matmul_op(const TensorImplPtr& x,
                                  const TensorImplPtr& w,
                                  const TensorImplPtr& scales,
                                  const TensorImplPtr& biases,
                                  bool transpose,
                                  int group_size,
                                  int bits) {
    ::mlx::core::array xa = gpu_array(x, "quantized_matmul");
    ::mlx::core::array wa = as_packed_u32(gpu_array(w, "quantized_matmul"));
    ::mlx::core::array sa = gpu_array(scales, "quantized_matmul");
    if (sa.dtype() != xa.dtype())
        sa = ::mlx::core::astype(sa, xa.dtype());
    std::optional<::mlx::core::array> ba = std::nullopt;
    if (biases) {
        ::mlx::core::array b = gpu_array(biases, "quantized_matmul");
        if (b.dtype() != xa.dtype())
            b = ::mlx::core::astype(b, xa.dtype());
        ba = std::move(b);
    }
    ::mlx::core::array out =
        ::mlx::core::quantized_matmul(xa, wa, sa, ba, transpose, group_size, bits);
    TensorImplPtr result = wrap(std::move(out), x->dtype());

    // 3.5 lucid.compile(): make this fused low-precision GEMM visible to the
    // active tracer.  It has no MPSGraph emitter AND its result depends on the
    // *live activation* x — so a compiled graph that silently baked it as a
    // trace-time constant would ignore new inputs (compiled(x2) == compiled(x1),
    // a silently-wrong output).  Recording it with a non-empty input list makes
    // the MpsBuilder find no emitter and abort the build, so the whole signature
    // falls back cleanly to eager (see MpsBuilder.mm's find_emitter gate).
    // dequantize is deliberately NOT recorded: its inputs are constant weights,
    // so baking its result is harmless and keeps that path compilable.
    //
    // The node must exist before wire_autograd, whose own on_op_io would
    // otherwise attach this op's I/O to the previous op's node.
    auto* tracer = ::lucid::compile::current_tracer();
    if (tracer)
        tracer->on_op_enter("quantized_matmul", Device::GPU, x->dtype(), result->shape());

    if (GradMode::is_enabled() && x->requires_grad()) {
        w->check_not_inference_saved("quantized_matmul");
        scales->check_not_inference_saved("quantized_matmul");
        if (biases)
            biases->check_not_inference_saved("quantized_matmul");
    }
    auto bwd = std::make_shared<QuantizedMatmulBackward>();
    bwd->saved_w_ = w->storage();
    bwd->saved_scales_ = scales->storage();
    bwd->w_shape_ = w->shape();
    bwd->scales_shape_ = scales->shape();
    if (biases) {
        bwd->saved_biases_ = biases->storage();
        bwd->biases_shape_ = biases->shape();
    }
    bwd->transpose_ = transpose;
    bwd->group_size_ = group_size;
    bwd->bits_ = bits;
    kernel::NaryKernel<QuantizedMatmulBackward, 1>::wire_autograd(std::move(bwd), {x}, result,
                                                                  false);

    // Last write wins: the frozen buffers join the input list wire_autograd recorded.
    if (tracer)
        tracer->on_op_io({x, w, scales, biases}, result);
    return result;
}

}  // namespace lucid
