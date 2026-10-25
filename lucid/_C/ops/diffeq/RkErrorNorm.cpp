// lucid/_C/ops/diffeq/RkErrorNorm.cpp
//
// Implements the fused embedded-error norm used by adaptive Runge-Kutta step
// control.  There is no backward node here on purpose: the result decides
// whether a step is accepted, which is control flow rather than a value the
// caller differentiates.  Returning a host double instead of a tensor makes
// that structural rather than a convention.
//
// Two paths, matching the engine's device split: Accelerate on the CPU
// stream, and a compiled MLX kernel on the GPU stream whose evaluation is
// forced exactly once by the download at the end.

#include "RkErrorNorm.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <mlx/compile.h>
#include <mlx/ops.h>

#include "../../backend/cpu/Blas.h"
#include "../../backend/cpu/Vdsp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Half.h"
#include "../../core/OpRegistry.h"
#include "../../core/OpSchema.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../utils/Contiguous.h"
#include "Operand.h"

namespace lucid {

namespace {

// Schema carrier.  The op has no backward node to hang ``schema_v1`` on, but
// every op is expected to be registered, so a tag struct owns it.
struct RkErrorNormOp {
    static const OpSchema schema_v1;
};

const OpSchema RkErrorNormOp::schema_v1{"rk_error_norm",  1, AmpPolicy::KeepInput,
                                        /*det=*/true,
                                        /*note=*/"",
                                        /*in_arity=*/-1,
                                        /*out_arity=*/1,
                                        /*stable_ins=*/{}};

LUCID_REGISTER_OP(RkErrorNormOp)

// Scratch the error vector fits in before the kernel reaches for the
// allocator.  An ODE state is usually small, and at those sizes a heap round
// trip is a large fraction of the whole call.
constexpr std::size_t kStackScratchDoubles = 256;

// Stage count from which materialising the error vector pays for itself.
//
// The two-pass form below trades one extra pass over `err` -- writing it,
// then reading it back -- for vectorised reads of the stages.  The trade is
// 2n against roughly 2.4x on s*n, so it breaks even just below three stages,
// which is what the measurements show: at n=4096 two terms cost 6.9us
// materialised against 6.0us in a single register-only pass, while fourteen
// cost 13.4us against 38.2us.
constexpr std::size_t kMaterialiseFromStages = 3;

// Element count from which an Accelerate call does more work than it costs to
// make.  A vector routine has a fixed entry cost of a few tens of
// nanoseconds; the register-only loop it replaces runs at roughly 0.7ns per
// element per stage, so the two meet around forty elements.  Below that the
// materialised form loses even at fourteen stages (measured: 1.99us against
// 1.86us at n=4), and an ODE state that small is common.
constexpr std::size_t kMinMaterialiseElements = 64;

// Reduce a materialised error vector against the tolerance.
//
// One pass over three contiguous arrays with no indirection, which is what
// lets the compiler vectorise it.
template <typename T>
double
reduce_ratio(const T* y0, const T* y1, const double* err, std::size_t n, double rtol, double atol) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a0 = std::abs(static_cast<double>(y0[i]));
        const double a1 = std::abs(static_cast<double>(y1[i]));
        const double ratio = err[i] / (atol + rtol * (a0 > a1 ? a0 : a1));
        acc += ratio * ratio;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

// One-pass form, for the stage counts that do not repay a materialised error
// vector.  The error estimate for element `i` is formed in a register and
// never stored; with one or two stages the compiler has few enough pointers
// to keep the loop worthwhile even without vectorising it.
template <typename T>
double error_norm_inline(const T* y0,
                         const T* y1,
                         const std::vector<const T*>& ks,
                         const std::vector<double>& scales,
                         std::size_t n,
                         double rtol,
                         double atol) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double err = 0.0;
        for (std::size_t j = 0; j < ks.size(); ++j)
            err += scales[j] * static_cast<double>(ks[j][i]);

        const double a0 = std::abs(static_cast<double>(y0[i]));
        const double a1 = std::abs(static_cast<double>(y1[i]));
        const double ratio = err / (atol + rtol * (a0 > a1 ? a0 : a1));
        acc += ratio * ratio;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

// CPU kernel, in two vectorised passes: build `err = sum_j scales[j]*ks[j]`
// through Accelerate, then reduce it against the tolerance.
//
// The obvious single pass -- stage index innermost, error formed in a
// register and never stored -- reads the stages through a table of pointers
// the compiler cannot prove non-aliasing, and loses the vector units for it.
// Materialising `err` costs a buffer and an extra pass over it and still wins
// by up to 5x (measured on an M1 Pro at n=65536, fourteen stages: 822us
// against 164us), because everything it does is vectorised.
//
// Accumulation stays in `double` whatever the input precision, as it did
// before: the error estimate is a difference of two nearly equal results, so
// it is exactly where cancellation bites.  A float32 input is therefore
// widened rather than accumulated in place.
template <typename T>
double error_norm_cpu(const T* y0,
                      const T* y1,
                      const std::vector<const T*>& ks,
                      const std::vector<double>& scales,
                      std::size_t n,
                      double rtol,
                      double atol) {
    if (ks.size() < kMaterialiseFromStages || n < kMinMaterialiseElements)
        return error_norm_inline<T>(y0, y1, ks, scales, n, rtol, atol);

    const bool on_stack = n <= kStackScratchDoubles;
    double stack_err[kStackScratchDoubles];
    // A float32 input needs a second buffer to widen each stage into; a
    // double one is read straight from the caller's storage.
    const bool widening = sizeof(T) != sizeof(double);
    double stack_tmp[kStackScratchDoubles];

    std::shared_ptr<std::byte[]> heap;
    double* err = stack_err;
    double* tmp = stack_tmp;
    if (!on_stack) {
        heap = allocate_aligned_bytes(n * sizeof(double) * (widening ? 2 : 1), Device::CPU);
        err = reinterpret_cast<double*>(heap.get());
        tmp = err + n;
    }

    for (std::size_t j = 0; j < ks.size(); ++j) {
        const double* term;
        if constexpr (sizeof(T) == sizeof(double)) {
            term = reinterpret_cast<const double*>(ks[j]);
        } else {
            backend::cpu::widen_f32_f64(reinterpret_cast<const float*>(ks[j]), tmp, n);
            term = tmp;
        }
        if (j == 0)
            backend::cpu::vsmul_f64(term, scales[0], err, n);
        else
            backend::cpu::daxpy(static_cast<int>(n), scales[j], term, err);
    }

    return reduce_ratio<T>(y0, y1, err, n, rtol, atol);
}

// The GPU stream's whole reduction, as one compiled Metal kernel.
//
// MLX does not fuse element-wise chains in eager mode, so the expression this
// replaces put two nodes per stage plus six more into the graph and launched
// every one of them.  Compiled, the stages are read once and reduced in
// place.  The same reasoning and the same shape of lambda as
// `rk_combine`'s -- capture-less, arity read from the input list, scales
// passed as inputs rather than baked in so one trace serves every tableau.
//
// Layout: [y0, y1, k_0 .. k_{m-1}, w_0 .. w_{m-1}, rtol, atol].
//
// It returns the *sum* of the squared ratios, not the root-mean-square: the
// mean would divide by an element count, and a shapeless trace is reused
// across shapes, so a count captured at trace time would be silently wrong
// for every later call with a different state size.  The division and the
// square root are host arithmetic on a scalar and cost nothing.
const std::function<std::vector<::mlx::core::array>(const std::vector<::mlx::core::array>&)>&
fused_error_norm() {
    namespace mx = ::mlx::core;
    static const std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled =
        mx::compile(
            [](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                const std::size_t m = (ins.size() - 4) / 2;
                mx::array err = mx::multiply(ins[2], ins[2 + m]);
                for (std::size_t j = 1; j < m; ++j)
                    err = mx::add(err, mx::multiply(ins[2 + j], ins[2 + m + j]));

                const mx::array& rtol = ins[ins.size() - 2];
                const mx::array& atol = ins[ins.size() - 1];
                mx::array tol = mx::add(
                    atol, mx::multiply(rtol, mx::maximum(mx::abs(ins[0]), mx::abs(ins[1]))));
                mx::array ratio = mx::divide(err, tol);
                return {mx::sum(mx::multiply(ratio, ratio), /*keepdims=*/false)};
            },
            /*shapeless=*/true);
    return compiled;
}

// Reads element 0 of a freshly downloaded one-element CPU buffer.
double read_scalar(const CpuStorage& cs, Dtype dtype) {
    if (dtype == Dtype::F64)
        return *reinterpret_cast<const double*>(cs.ptr.get());
    return static_cast<double>(*reinterpret_cast<const float*>(cs.ptr.get()));
}

double error_norm_half_cpu(const CpuStorage& c0,
                           const CpuStorage& c1,
                           const std::vector<TensorImplPtr>& ks_c,
                           const std::vector<double>& scales,
                           Dtype dtype,
                           std::size_t n,
                           double rtol,
                           double atol) {
    const bool is_bf16 = dtype == Dtype::BF16;
    const std::size_t k_count = ks_c.size();
    auto scratch = allocate_aligned_bytes(n * (k_count + 2) * sizeof(float), Device::CPU);
    auto* y0_f = reinterpret_cast<float*>(scratch.get());
    auto* y1_f = y0_f + n;
    backend::detail::half_or_bfloat16_to_float_bulk(
        reinterpret_cast<const std::uint16_t*>(c0.ptr.get()), y0_f, n, is_bf16);
    backend::detail::half_or_bfloat16_to_float_bulk(
        reinterpret_cast<const std::uint16_t*>(c1.ptr.get()), y1_f, n, is_bf16);
    std::vector<const float*> kp;
    kp.reserve(k_count);
    for (std::size_t j = 0; j < k_count; ++j) {
        float* kj = y1_f + (j + 1) * n;
        const auto& ck = std::get<CpuStorage>(ks_c[j]->storage());
        backend::detail::half_or_bfloat16_to_float_bulk(
            reinterpret_cast<const std::uint16_t*>(ck.ptr.get()), kj, n, is_bf16);
        kp.push_back(kj);
    }
    return error_norm_cpu<float>(y0_f, y1_f, kp, scales, n, rtol, atol);
}

double error_norm_dispatch_cpu(const TensorImplPtr& y0_c,
                               const TensorImplPtr& y1_c,
                               const std::vector<TensorImplPtr>& ks_c,
                               const std::vector<double>& scales,
                               Dtype dtype,
                               std::size_t n,
                               double rtol,
                               double atol) {
    const auto& c0 = std::get<CpuStorage>(y0_c->storage());
    const auto& c1 = std::get<CpuStorage>(y1_c->storage());
    if (is_half_float(dtype))
        return error_norm_half_cpu(c0, c1, ks_c, scales, dtype, n, rtol, atol);
    if (dtype == Dtype::F64) {
        std::vector<const double*> kp;
        kp.reserve(ks_c.size());
        for (const auto& k : ks_c)
            kp.push_back(
                reinterpret_cast<const double*>(std::get<CpuStorage>(k->storage()).ptr.get()));
        return error_norm_cpu<double>(reinterpret_cast<const double*>(c0.ptr.get()),
                                      reinterpret_cast<const double*>(c1.ptr.get()), kp, scales, n,
                                      rtol, atol);
    }
    std::vector<const float*> kp;
    kp.reserve(ks_c.size());
    for (const auto& k : ks_c)
        kp.push_back(reinterpret_cast<const float*>(std::get<CpuStorage>(k->storage()).ptr.get()));
    return error_norm_cpu<float>(reinterpret_cast<const float*>(c0.ptr.get()),
                                 reinterpret_cast<const float*>(c1.ptr.get()), kp, scales, n, rtol,
                                 atol);
}

double error_norm_gpu(const TensorImplPtr& y0_c,
                      const TensorImplPtr& y1_c,
                      const std::vector<TensorImplPtr>& ks_c,
                      const std::vector<double>& scales,
                      Dtype dtype,
                      std::size_t n,
                      double rtol,
                      double atol) {
    namespace mx = ::mlx::core;
    const bool widen_half = is_half_float(dtype);
    const Dtype accum_dt = widen_half ? Dtype::F32 : dtype;
    const mx::Dtype mdt =
        widen_half ? mx::float32 : std::get<GpuStorage>(y0_c->storage()).arr->dtype();
    auto as_accum = [&](const TensorImplPtr& t) -> mx::array {
        const mx::array& arr = *std::get<GpuStorage>(t->storage()).arr;
        return widen_half ? mx::astype(arr, mdt) : arr;
    };

    std::vector<mx::array> ins;
    ins.reserve(2 * ks_c.size() + 4);
    ins.push_back(as_accum(y0_c));
    ins.push_back(as_accum(y1_c));
    for (const auto& k : ks_c)
        ins.push_back(as_accum(k));
    for (const double scale : scales)
        ins.push_back(mx::array(scale, mdt));
    ins.push_back(mx::array(rtol, mdt));
    ins.push_back(mx::array(atol, mdt));

    GpuStorage gs = gpu::wrap_mlx_array(std::move(fused_error_norm()(ins)[0]), accum_dt);
    const CpuStorage host = gpu::download_gpu_to_cpu(gs, Shape{});
    return std::sqrt(read_scalar(host, accum_dt) / static_cast<double>(n));
}

}  // namespace

// Validate the operands against ``y0``, materialise any views, then dispatch
// to the device-appropriate kernel.  Terms whose folded scale is exactly zero
// contribute nothing and are skipped; when none survive the error estimate is
// identically zero, so the ratio is zero and the step is always accepted.
double rk_error_norm_op(const TensorImplPtr& y0,
                        const TensorImplPtr& y1,
                        const std::vector<TensorImplPtr>& ks,
                        const std::vector<double>& coeffs,
                        double dt,
                        double rtol,
                        double atol) {
    Validator::input(y0, "rk_error_norm.y0").non_null();
    Validator::input(y1, "rk_error_norm.y1").non_null();
    if (ks.size() != coeffs.size())
        ErrorBuilder("rk_error_norm").fail("ks and coeffs must have the same length");

    const Dtype dtype = y0->dtype();
    const Device device = y0->device();
    const Shape shape = y0->shape();

    if (!is_floating_point(dtype))
        throw NotImplementedError("rk_error_norm: dtype " + std::string(dtype_name(dtype)) +
                                  " is not supported; promote to float32 or float64 first");

    const diffeq::OperandSpec spec = diffeq::OperandSpec::from(y0, "rk_error_norm");
    diffeq::check_operand(y1, "rk_error_norm.y1", spec);
    for (const auto& k : ks)
        diffeq::check_operand(k, "rk_error_norm.ks", spec);

    OpScopeFull scope{"rk_error_norm", device, dtype, shape};

    std::vector<double> scales;
    std::vector<std::size_t> live;
    scales.reserve(ks.size());
    live.reserve(ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double scale = dt * coeffs[i];
        if (scale == 0.0)
            continue;
        scales.push_back(scale);
        live.push_back(i);
    }
    if (live.empty())
        return 0.0;

    const std::size_t n = shape_numel(shape);
    if (n == 0)
        return 0.0;

    // Views have to be materialised for both paths.  No gradient is wired:
    // this value is control flow, so attaching a graph here would build nodes
    // that nothing can ever consume.
    NoGradGuard no_grad;
    const TensorImplPtr y0_c = y0->is_contiguous() ? y0 : contiguous_op(y0);
    const TensorImplPtr y1_c = y1->is_contiguous() ? y1 : contiguous_op(y1);
    std::vector<TensorImplPtr> ks_c;
    ks_c.reserve(live.size());
    for (const std::size_t idx : live) {
        const TensorImplPtr& k = ks[idx];
        ks_c.push_back(k->is_contiguous() ? k : contiguous_op(k));
    }

    if (device == Device::CPU)
        return error_norm_dispatch_cpu(y0_c, y1_c, ks_c, scales, dtype, n, rtol, atol);
    return error_norm_gpu(y0_c, y1_c, ks_c, scales, dtype, n, rtol, atol);
}

}  // namespace lucid
