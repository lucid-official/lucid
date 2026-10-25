// lucid/_C/ops/diffeq/RkCombine.cpp
//
// Implements the fused Runge-Kutta stage combination and its backward node.
//
//   RkCombineBackward — routes the output gradient straight to `y0` and a
//                       `dt * coeffs[i]`-scaled copy to each stage input.
//
// The node derives from VariadicKernel because the stage count is a runtime
// property of the Butcher tableau, so the fixed-arity AutogradNode<Derived,N>
// (which stores its edges in a compile-time std::array) cannot express it.
//
// The forward is one pass per device family rather than a chain through the
// backend dispatcher.  Written as a chain the op is fused only at the Python
// level: it still costs two backend calls and one temporary per stage, so its
// price grows with the stage count -- which is exactly backwards, since the
// methods with the most stages are the ones that call it most often.  dopri8
// runs fourteen stages and invokes this fourteen times a step.

#include "RkCombine.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/compile.h>
#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/cpu/Blas.h"
#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/OpSchema.h"
#include "../../core/PhantomMode.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/VariadicKernel.h"
#include "../bfunc/Mul.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Contiguous.h"
#include "Operand.h"

namespace lucid {

namespace {

// Backward node for rk_combine_op.
//
// Invariants:
//   scales_ — `dt * coeffs[i]` folded into a single factor per stage input,
//             computed once at forward time.  Its length equals the number
//             of `ks` inputs, so `scales_.size() + 1` gradients are returned.
//
// Backward formula: the forward is affine in every input, so the Jacobian is
// constant.  `dL/dy0 = g` (a clone, because the engine may not alias the same
// buffer into several gradient slots) and `dL/dk_i = dt * coeffs[i] * g`.
// Gradients are returned in input order — `y0` first, then the stages.
class RkCombineBackward : public kernel::VariadicKernel<RkCombineBackward> {
public:
    static const OpSchema schema_v1;

    std::vector<double> scales_;

    std::string node_name() const override { return std::string(schema_v1.name); }

    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(out_shape_);
        std::vector<Storage> grads;
        grads.reserve(scales_.size() + 1);
        grads.push_back(clone_storage(grad_out, n, dtype_, device_));
        for (const double scale : scales_)
            grads.push_back(mul_scalar_storage(grad_out, scale, n, dtype_, device_));
        return grads;
    }

    // Graph-recording backward, so a fused step stays as differentiable as
    // the unfused `y + dt * c * k` spelling it replaces — without this the
    // fusion would quietly cost the caller second-order gradients.
    //
    // The scale is materialised as a 0-D tensor and applied with `mul_op`
    // rather than a raw scalar kernel: that is the same route Python's
    // `tensor * scalar` takes, so the resulting node is differentiable in
    // turn.  `dL/dy0` is the identity, so `grad_out` is forwarded as-is.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        std::vector<TensorImplPtr> grads;
        grads.reserve(scales_.size() + 1);
        grads.push_back(grad_out);
        for (const double scale : scales_) {
            auto factor = full_op(Shape{}, scale, grad_out->dtype(), grad_out->device());
            grads.push_back(mul_op(grad_out, factor));
        }
        return grads;
    }
};

const OpSchema RkCombineBackward::schema_v1{"rk_combine",     1, AmpPolicy::Promote,
                                            /*det=*/true,
                                            /*note=*/"",
                                            /*in_arity=*/-1,
                                            /*out_arity=*/1,
                                            /*stable_ins=*/{}};

LUCID_REGISTER_OP(RkCombineBackward)

// `out = y0 + sum_j scales[j] * ks[j]` on the CPU stream, one Accelerate call
// per term and no temporaries.
//
// Accelerate rather than a hand-written loop over all the stages at once: a
// loop with the stage index innermost reads through a table of pointers the
// compiler cannot prove non-aliasing, which costs it the vectorisation.
// Measured on an M1 Pro at n=4096, fourteen stages: 45.3us for the hand loop
// against 7.0us for the same arithmetic through Accelerate.  The extra passes
// over `out` that one call per term implies are cheaper than losing the
// vector units.
//
// The first term goes through the three-buffer `vsma`, reading `y0` and
// writing `out`, so nothing has to seed `out` with a copy of `y0` first; the
// rest accumulate in place with `axpy`.  Seeding with a copy instead is
// faster once the state stops fitting in cache -- `vDSP_vsma` has about half
// the throughput of `cblas_axpy` at n=65536 -- but it costs a whole extra
// pass below that, and there it is the difference between beating the chained
// spelling and losing to it (n=4096, two terms: 3.8us against 6.2us for the
// copy and 4.7us for the chain).  Small states are the common case for an
// ODE, so the copy stays out.
template <typename T>
void combine_cpu(const T* y0,
                 const std::vector<const T*>& ks,
                 const std::vector<double>& scales,
                 T* out,
                 std::size_t n) {
    if constexpr (sizeof(T) == sizeof(double)) {
        backend::cpu::vsma_f64(ks[0], scales[0], y0, out, n);
        for (std::size_t j = 1; j < ks.size(); ++j)
            backend::cpu::daxpy(static_cast<int>(n), scales[j], ks[j], out);
    } else {
        backend::cpu::vsma_f32(ks[0], static_cast<float>(scales[0]), y0, out, n);
        for (std::size_t j = 1; j < ks.size(); ++j)
            backend::cpu::saxpy(static_cast<int>(n), static_cast<float>(scales[j]), ks[j], out);
    }
}

// Typed view of a contiguous CPU tensor's buffer.
template <typename T>
const T* cpu_data(const TensorImplPtr& t) {
    return reinterpret_cast<const T*>(std::get<CpuStorage>(t->storage()).ptr.get());
}

// The GPU stream's whole combination, as a single fused Metal kernel.
//
// MLX does not fuse element-wise chains in eager mode -- that is what compile
// is for -- so the chained spelling pays a kernel launch and a round trip to
// memory for every multiply and every add.  Compiled, the terms are read once
// and the result written once, whatever the stage count.
//
// Two spellings were measured against it and both lose.  The chain loses on
// launches, which dominate whenever the state is small: at n=4 it costs the
// same as at n=4096, and grows linearly in the stage count either way.
// Concatenating the terms into one array and reducing that is flat in the
// stage count but materialises an extra copy of every term, so it wins only
// while the state is small enough for launches to matter and loses badly
// once it is not (n=262144, fourteen stages: 441us against the chain's 377).
// Compiling wins at every state size and stage count measured -- 154us for
// that same case, and 29us against the chain's 33 at one stage.
//
// The scales arrive as inputs rather than as captured constants so that one
// trace serves every Butcher row; baking them in would compile a fresh kernel
// per tableau row per step size.  The lambda is capture-less (compile needs a
// function-pointer-convertible one) and reads its arity from the input list,
// so a single static covers every stage count -- MLX traces once per distinct
// input signature and reuses it after that.
const std::function<std::vector<::mlx::core::array>(const std::vector<::mlx::core::array>&)>&
fused_combine() {
    namespace mx = ::mlx::core;
    static const std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled =
        mx::compile(
            [](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                // ins = [y0, k_0 .. k_{m-1}, w_0 .. w_{m-1}]
                const std::size_t m = (ins.size() - 1) / 2;
                mx::array acc = ins[0];
                for (std::size_t j = 0; j < m; ++j)
                    acc = mx::add(acc, mx::multiply(ins[1 + j], ins[1 + m + j]));
                return {acc};
            },
            /*shapeless=*/true);
    return compiled;
}

Storage combine_half_cpu(const TensorImplPtr& y0,
                         const std::vector<TensorImplPtr>& inputs,
                         const std::vector<std::size_t>& live,
                         const std::vector<double>& scales,
                         Dtype dtype,
                         std::size_t n) {
    const bool is_bf16 = dtype == Dtype::BF16;
    const std::size_t k_count = live.size();
    auto scratch = allocate_aligned_bytes(n * (k_count + 2) * sizeof(float), Device::CPU);
    auto* y0_f = reinterpret_cast<float*>(scratch.get());
    auto* out_f = y0_f + n;
    backend::detail::half_or_bfloat16_to_float_bulk(cpu_data<std::uint16_t>(y0), y0_f, n, is_bf16);
    std::vector<const float*> kp;
    kp.reserve(k_count);
    for (std::size_t j = 0; j < k_count; ++j) {
        float* kj = out_f + (j + 1) * n;
        backend::detail::half_or_bfloat16_to_float_bulk(cpu_data<std::uint16_t>(inputs[live[j]]),
                                                        kj, n, is_bf16);
        kp.push_back(kj);
    }
    combine_cpu<float>(y0_f, kp, scales, out_f, n);
    const std::size_t nbytes = n * dtype_size(dtype);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    backend::detail::float_to_half_or_bfloat16_bulk(
        out_f, reinterpret_cast<std::uint16_t*>(ptr.get()), n, is_bf16);
    return Storage{CpuStorage{ptr, nbytes, dtype}};
}

// `y0 + sum_j scales[j] * inputs[live[j]]`, as one pass per device family.
//
// `live` indexes `inputs`, whose slot 0 is `y0` itself; only stages with a
// non-zero folded scale appear.  With none the answer is a copy of `y0`.
Storage combine_storage(const TensorImplPtr& y0,
                        const std::vector<TensorImplPtr>& inputs,
                        const std::vector<std::size_t>& live,
                        const std::vector<double>& scales,
                        const Shape& shape,
                        Dtype dtype,
                        Device device,
                        std::size_t n) {
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dtype, device);
    if (live.empty() || n == 0)
        return clone_storage(y0->storage(), n, dtype, device);

    if (device == Device::CPU) {
        if (is_half_float(dtype))
            return combine_half_cpu(y0, inputs, live, scales, dtype, n);
        if (dtype != Dtype::F32 && dtype != Dtype::F64)
            throw NotImplementedError("rk_combine: dtype " + std::string(dtype_name(dtype)) +
                                      " is not supported on the CPU stream");
        // One buffer for the whole call, where the chained spelling allocated
        // one per stage.
        const std::size_t nbytes = n * dtype_size(dtype);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        if (dtype == Dtype::F64) {
            std::vector<const double*> kp;
            kp.reserve(live.size());
            for (const std::size_t idx : live)
                kp.push_back(cpu_data<double>(inputs[idx]));
            combine_cpu<double>(cpu_data<double>(y0), kp, scales,
                                reinterpret_cast<double*>(ptr.get()), n);
        } else {
            std::vector<const float*> kp;
            kp.reserve(live.size());
            for (const std::size_t idx : live)
                kp.push_back(cpu_data<float>(inputs[idx]));
            combine_cpu<float>(cpu_data<float>(y0), kp, scales, reinterpret_cast<float*>(ptr.get()),
                               n);
        }
        return Storage{CpuStorage{ptr, nbytes, dtype}};
    }

    // GPU: hand the whole combination to one compiled kernel.  See
    // `fused_combine` for what that replaces and what else was tried.
    namespace mx = ::mlx::core;
    const mx::Dtype mdt = std::get<GpuStorage>(y0->storage()).arr->dtype();

    std::vector<mx::array> ins;
    ins.reserve(2 * live.size() + 1);
    ins.push_back(*std::get<GpuStorage>(y0->storage()).arr);
    for (const std::size_t idx : live)
        ins.push_back(*std::get<GpuStorage>(inputs[idx]->storage()).arr);
    for (const double scale : scales)
        ins.push_back(mx::array(scale, mdt));

    return Storage{gpu::wrap_mlx_array(std::move(fused_combine()(ins)[0]), dtype)};
}

}  // namespace

// Validate that every stage tensor matches `y0` in shape, dtype, and device,
// then evaluate `y0 + dt * sum_i coeffs[i] * ks[i]` in one pass and attach
// RkCombineBackward.  Stage terms whose folded scale is exactly zero
// contribute nothing and are skipped — Butcher rows are strictly lower
// triangular, so the zeros are the common case rather than the corner.
TensorImplPtr rk_combine_op(const TensorImplPtr& y0,
                            const std::vector<TensorImplPtr>& ks,
                            const std::vector<double>& coeffs,
                            double dt) {
    Validator::input(y0, "rk_combine.y0").non_null();
    if (ks.size() != coeffs.size())
        ErrorBuilder("rk_combine").fail("ks and coeffs must have the same length");

    const Dtype dtype = y0->dtype();
    const Device device = y0->device();
    const Shape shape = y0->shape();

    const diffeq::OperandSpec spec = diffeq::OperandSpec::from(y0, "rk_combine");
    for (std::size_t i = 0; i < ks.size(); ++i)
        diffeq::check_operand(ks[i], "rk_combine.ks[" + std::to_string(i) + "]", spec);

    OpScopeFull scope{"rk_combine", device, dtype, shape};
    scope.set_attr("stages", static_cast<std::int64_t>(ks.size()));

    // The backend arithmetic below is stride-agnostic, so any view input has
    // to be materialised first.  contiguous_op is a no-op for tensors that
    // already are contiguous, and wires its own backward when they are not.
    const TensorImplPtr y0_c = y0->is_contiguous() ? y0 : contiguous_op(y0);
    std::vector<TensorImplPtr> inputs;
    inputs.reserve(ks.size() + 1);
    inputs.push_back(y0_c);
    for (const auto& k : ks)
        inputs.push_back(k->is_contiguous() ? k : contiguous_op(k));

    // Every stage keeps an entry in `scales_` so the backward node stays
    // aligned with the input list, but only the non-zero ones take part in
    // the arithmetic: Butcher rows are strictly lower triangular, so the
    // zeros are the common case rather than the corner.
    std::vector<double> scales;
    std::vector<double> live_scales;
    std::vector<std::size_t> live;
    scales.reserve(ks.size());
    live_scales.reserve(ks.size());
    live.reserve(ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double scale = dt * coeffs[i];
        scales.push_back(scale);
        if (scale == 0.0)
            continue;
        live_scales.push_back(scale);
        live.push_back(i + 1);  // index into `inputs`, whose slot 0 is y0
    }

    const std::size_t n = shape_numel(shape);
    Storage out_storage = combine_storage(y0_c, inputs, live, live_scales, shape, dtype, device, n);
    auto out = helpers::fresh(std::move(out_storage), shape, dtype, device);

    auto bwd = std::make_shared<RkCombineBackward>();
    bwd->scales_ = std::move(scales);
    kernel::VariadicKernel<RkCombineBackward>::wire_autograd(std::move(bwd), inputs, out,
                                                             /*save_ins=*/false);
    return out;
}

}  // namespace lucid
