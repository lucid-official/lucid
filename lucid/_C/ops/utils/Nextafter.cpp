// lucid/_C/ops/utils/Nextafter.cpp
//
// IEEE-754 next-representable-float computation.  Two paths:
//   * CPU (F32 or F64): per-element std::nextafter loop.
//   * GPU (F32 only): pure MLX bit-manipulation pipeline using view()
//     to reinterpret the float32 buffer as int32, applying the
//     IEEE-754 next-toward rule via where + add, and viewing back.
//     F64 is not supported on Metal (MLX limitation) — F64 GPU inputs
//     fall through the existing CPU round-trip path.
//
// Both inputs must share the same float dtype and shape.

#include "Nextafter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/BinaryKernelLoad.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Half.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/BinaryKernel.h"  // detail::broadcast_shapes
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Mul.h"
#include "../gfunc/Gfunc.h"  // zeros_like_op
#include "../ufunc/Reductions.h"
#include "Layout.h"  // broadcast_to_op
#include "View.h"    // reshape_op
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;

CpuStorage to_cpu(const TensorImplPtr& a) {
    return backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());
}

Storage to_device_storage(CpuStorage&& cpu, Device target_device, const Shape& shape) {
    if (target_device == Device::GPU && cpu.dtype != Dtype::F64) {
        return backend::Dispatcher::for_device(Device::GPU).from_cpu(cpu, shape);
    }
    return Storage{std::move(cpu)};
}

// Element-wise std::nextafter loop, templated over the float type.
template <typename T>
void run_nextafter(const T* a, const T* b, T* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = std::nextafter(a[i], b[i]);
}

// Engine-side IEEE-754 next-toward-b for F32 on Metal/MLX.  The algorithm
// is a pure-elementwise bit twiddle:
//
//   * If either input is NaN → NaN.
//   * If a == b (including ±0 == ±0) → b.
//   * If a == 0 (and b is finite, non-equal): return ±smallest positive
//     subnormal with sign matching b.
//   * Otherwise: convert a to its int32 bit pattern, increment the bits
//     toward b (decrement when a is negative), bitcast back.
//
// Returns the result as a GpuStorage (F32, GPU device).
GpuStorage nextafter_gpu_f32(const GpuStorage& a, const GpuStorage& b) {
    namespace mx = ::mlx::core;
    auto& af = *a.arr;
    auto& bf = *b.arr;

    // Bit reinterpretation: f32 ↔ int32.
    auto a_bits = mx::view(af, mx::int32);
    auto b_bits = mx::view(bf, mx::int32);

    auto zero_f = mx::array(0.0f, mx::float32);
    auto one_i = mx::array(static_cast<std::int32_t>(1), mx::int32);
    auto neg_one_i = mx::array(static_cast<std::int32_t>(-1), mx::int32);

    // direction = +1 when b > a, else -1.  delta = direction when a > 0,
    // else -direction (negative-a bits move opposite the float ordering).
    auto direction = mx::where(mx::greater(bf, af), one_i, neg_one_i);
    auto a_positive = mx::greater(af, zero_f);
    auto delta = mx::where(a_positive, direction, mx::negative(direction));

    auto general = mx::add(a_bits, delta);

    // a == 0 special case: smallest +subnormal toward positive b, or
    // smallest -subnormal toward negative b.
    auto smallest_pos = mx::array(static_cast<std::int32_t>(1), mx::int32);
    auto smallest_neg = mx::array(static_cast<std::int32_t>(0x80000001u), mx::int32);
    auto a_zero = mx::equal(af, zero_f);
    auto b_pos = mx::greater(bf, zero_f);
    auto zero_branch = mx::where(b_pos, smallest_pos, smallest_neg);

    auto step = mx::where(a_zero, zero_branch, general);

    // a == b → b's bits (preserves sign-of-zero).
    auto eq = mx::equal(af, bf);
    step = mx::where(eq, b_bits, step);

    // NaN propagation: f != f.
    auto a_nan = mx::not_equal(af, af);
    auto b_nan = mx::not_equal(bf, bf);
    auto any_nan = mx::logical_or(a_nan, b_nan);
    auto nan_bits =
        mx::view(mx::array(std::numeric_limits<float>::quiet_NaN(), mx::float32), mx::int32);
    step = mx::where(any_nan, nan_bits, step);

    auto out = mx::view(step, mx::float32);
    return gpu::wrap_mlx_array(std::move(out), Dtype::F32);
}

}  // namespace

TensorImplPtr nextafter_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    Validator::input(a_in, "nextafter.a").non_null();
    Validator::input(b_in, "nextafter.b").non_null();

    if (a_in->dtype() != b_in->dtype())
        throw DtypeMismatch(std::string(dtype_name(a_in->dtype())),
                            std::string(dtype_name(b_in->dtype())), "nextafter");

    auto [a, b] = detail::align_binary_devices(a_in, b_in, "nextafter");
    if (a->dtype() != Dtype::F32 && a->dtype() != Dtype::F64)
        ErrorBuilder("nextafter").fail("dtype must be F32 or F64");

    // Broadcast like every other elementwise op.  Only materialised when
    // the shapes actually differ, so a same-shape call — which is every
    // call that worked before — reaches the kernel untouched on either
    // device and adds nothing to a compile trace.
    if (a->shape() != b->shape()) {
        const Shape out = detail::broadcast_shapes(a->shape(), b->shape());
        if (a->shape() != out)
            a = broadcast_to_op(a, out);
        if (b->shape() != out)
            b = broadcast_to_op(b, out);
    }

    OpScopeFull scope{"nextafter", a->device(), a->dtype(), a->shape()};

    if (PhantomMode::is_enabled()) [[unlikely]] {
        auto result = fresh(phantom_storage(a->shape(), a->dtype(), a->device()), a->shape(),
                            a->dtype(), a->device());
        kernel::NaryKernel<NextafterBackward, 2>::wire_autograd(
            std::make_shared<NextafterBackward>(), {a, b}, result, false);
        return result;
    }

    // Metal/F32 fast path: bit-twiddle on MLX, no CPU round-trip.
    if (a->device() == Device::GPU && a->dtype() == Dtype::F32) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        const auto& gb = std::get<GpuStorage>(b->storage());
        auto result =
            fresh(Storage{nextafter_gpu_f32(ga, gb)}, a->shape(), a->dtype(), Device::GPU);
        // ``wire_autograd`` also records the op's operands on an active trace;
        // without that ``lucid.compile`` sees no inputs and refuses the graph.
        kernel::NaryKernel<NextafterBackward, 2>::wire_autograd(
            std::make_shared<NextafterBackward>(), {a, b}, result, false);
        return result;
    }

    // CPU fallback (and F64 on either device — MLX doesn't support F64).
    const auto ca = to_cpu(a);
    const auto cb = to_cpu(b);
    const std::size_t n = shape_numel(a->shape());
    auto out = allocate_cpu(a->shape(), a->dtype());

    if (a->dtype() == Dtype::F32) {
        run_nextafter(reinterpret_cast<const float*>(ca.ptr.get()),
                      reinterpret_cast<const float*>(cb.ptr.get()),
                      reinterpret_cast<float*>(out.ptr.get()), n);
    } else {
        run_nextafter(reinterpret_cast<const double*>(ca.ptr.get()),
                      reinterpret_cast<const double*>(cb.ptr.get()),
                      reinterpret_cast<double*>(out.ptr.get()), n);
    }

    Storage final_storage = to_device_storage(std::move(out), a->device(), a->shape());
    auto result = fresh(std::move(final_storage), a->shape(), a->dtype(), a->device());
    kernel::NaryKernel<NextafterBackward, 2>::wire_autograd(std::make_shared<NextafterBackward>(),
                                                            {a, b}, result, false);
    return result;
}

const OpSchema NextafterBackward::schema_v1{"nextafter", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> NextafterBackward::apply(Storage grad_out) {
    Storage zeros = backend::Dispatcher::for_device(device_).zeros(out_shape_, dtype_);
    return {std::move(grad_out), std::move(zeros)};
}

std::vector<TensorImplPtr> NextafterBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {grad_out, zeros_like_op(grad_out)};
}

LUCID_REGISTER_OP(NextafterBackward)

TensorImplPtr bitcast_op(const TensorImplPtr& a, Dtype dtype) {
    Validator::input(a, "bitcast.a").non_null();
    if (dtype_size(a->dtype()) != dtype_size(dtype))
        ErrorBuilder("bitcast").invalid_argument("bitcast requires equal element byte widths");
    if (a->dtype() == dtype)
        return a;
    const auto& shape = a->shape();
    OpScopeFull scope{"bitcast", a->device(), dtype, shape};
    TensorImplPtr out_impl;
    if (a->is_phantom()) {
        out_impl = fresh(Storage{PhantomStorage{0, dtype, a->device()}}, shape, dtype, a->device());
    } else if (a->device() == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        auto out = ::mlx::core::view(*ga.arr, gpu::to_mlx_dtype(dtype));
        out_impl =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dtype)}, shape, dtype, Device::GPU);
    } else {
        CpuStorage cpu = to_cpu(a);
        cpu.dtype = dtype;
        out_impl = fresh(Storage{std::move(cpu)}, shape, dtype, Device::CPU);
    }
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out_impl);
        trc->on_op_attr("dst_dtype", static_cast<std::int64_t>(static_cast<int>(dtype)));
    }
    return out_impl;
}

namespace {

template <typename T>
void run_fma_float(const void* a, const void* b, const void* c, void* dst, std::size_t n) {
    const auto* pa = reinterpret_cast<const T*>(a);
    const auto* pb = reinterpret_cast<const T*>(b);
    const auto* pc = reinterpret_cast<const T*>(c);
    auto* pd = reinterpret_cast<T*>(dst);
    for (std::size_t i = 0; i < n; ++i)
        pd[i] = std::fma(pa[i], pb[i], pc[i]);
}

void run_fma_f16(const void* a, const void* b, const void* c, void* dst, std::size_t n) {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    const auto* pa = reinterpret_cast<const _Float16*>(a);
    const auto* pb = reinterpret_cast<const _Float16*>(b);
    const auto* pc = reinterpret_cast<const _Float16*>(c);
    auto* pd = reinterpret_cast<_Float16*>(dst);
    for (std::size_t i = 0; i < n; ++i)
        pd[i] = __builtin_elementwise_fma(pa[i], pb[i], pc[i]);
#else
    const auto* pa = reinterpret_cast<const std::uint16_t*>(a);
    const auto* pb = reinterpret_cast<const std::uint16_t*>(b);
    const auto* pc = reinterpret_cast<const std::uint16_t*>(c);
    auto* pd = reinterpret_cast<std::uint16_t*>(dst);
    for (std::size_t i = 0; i < n; ++i) {
        const double fa = backend::detail::half_bits_to_float(pa[i]);
        const double fb = backend::detail::half_bits_to_float(pb[i]);
        const double fc = backend::detail::half_bits_to_float(pc[i]);
        pd[i] = backend::detail::float_to_half_bits(static_cast<float>(std::fma(fa, fb, fc)));
    }
#endif
}

void run_fma_bf16(const void* a, const void* b, const void* c, void* dst, std::size_t n) {
    const auto* pa = reinterpret_cast<const std::uint16_t*>(a);
    const auto* pb = reinterpret_cast<const std::uint16_t*>(b);
    const auto* pc = reinterpret_cast<const std::uint16_t*>(c);
    auto* pd = reinterpret_cast<std::uint16_t*>(dst);
    for (std::size_t i = 0; i < n; ++i) {
        const float fa = backend::detail::bfloat16_bits_to_float(pa[i]);
        const float fb = backend::detail::bfloat16_bits_to_float(pb[i]);
        const float fc = backend::detail::bfloat16_bits_to_float(pc[i]);
        float s = std::fma(fa, fb, fc);
        std::uint32_t s_bits = 0;
        std::memcpy(&s_bits, &s, sizeof(float));
        if ((s_bits & 0x7f800000u) != 0x7f800000u && (s_bits & 0xffffu) == 0x8000u) {
            const float p = fa * fb;
            const float err = (std::fabs(p) >= std::fabs(fc)) ? (std::fma(fa, fb, -s) + fc)
                                                              : std::fma(fa, fb, fc - s);
            if (err != 0.0f) {
                s_bits = ((err > 0.0f) == (s > 0.0f)) ? (s_bits | 1u) : (s_bits - 1u);
                std::memcpy(&s, &s_bits, sizeof(float));
            }
        }
        pd[i] = backend::detail::float_to_bfloat16_bits(s);
    }
}

CpuStorage cpu_at_shape(const TensorImplPtr& t, const Shape& out_shape, Dtype dt) {
    CpuStorage raw = to_cpu(t);
    return (t->shape() == out_shape) ? std::move(raw)
                                     : detail::broadcast_cpu(raw, t->shape(), out_shape, dt);
}

CpuStorage fma_cpu(const TensorImplPtr& a,
                   const TensorImplPtr& b,
                   const TensorImplPtr& c,
                   const Shape& out_shape,
                   Dtype dt) {
    const CpuStorage ca = cpu_at_shape(a, out_shape, dt);
    const CpuStorage cb = cpu_at_shape(b, out_shape, dt);
    const CpuStorage cc = cpu_at_shape(c, out_shape, dt);
    const std::size_t n = shape_numel(out_shape);
    CpuStorage out = allocate_cpu(out_shape, dt);
    if (dt == Dtype::F32)
        run_fma_float<float>(ca.ptr.get(), cb.ptr.get(), cc.ptr.get(), out.ptr.get(), n);
    else if (dt == Dtype::F64)
        run_fma_float<double>(ca.ptr.get(), cb.ptr.get(), cc.ptr.get(), out.ptr.get(), n);
    else if (dt == Dtype::F16)
        run_fma_f16(ca.ptr.get(), cb.ptr.get(), cc.ptr.get(), out.ptr.get(), n);
    else if (dt == Dtype::BF16)
        run_fma_bf16(ca.ptr.get(), cb.ptr.get(), cc.ptr.get(), out.ptr.get(), n);
    else
        ErrorBuilder("fma").not_implemented("dtype not supported");
    return out;
}

GpuStorage fma_gpu(const GpuStorage& a,
                   const GpuStorage& b,
                   const GpuStorage& c,
                   const Shape& out_shape,
                   Dtype dt) {
    namespace mx = ::mlx::core;
    const auto mlx_shape = gpu::to_mlx_shape(out_shape);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    const std::size_t n = shape_numel(out_shape);
    if (n == 0)
        return gpu::wrap_mlx_array(mx::zeros(mlx_shape, mlx_dt), dt);
    const mx::Shape k_shape = mlx_shape.empty() ? mx::Shape{1} : mlx_shape;
    auto at_shape = [&](const GpuStorage& s) {
        const auto& arr = *s.arr;
        return (arr.shape() == k_shape) ? arr : mx::broadcast_to(arr, k_shape);
    };
    static const auto kernel =
        mx::fast::metal_kernel("lucid_fma", {"a", "b", "c"}, {"out"},
                               std::string(backend::unary_elem_index_src()) +
                                   "out[elem] = fma(a[elem], b[elem], c[elem]);\n",
                               "", true, false, mx::CompileOptions{});
    static const auto kernel_bf16 =
        mx::fast::metal_kernel("lucid_fma_bf16", {"a", "b", "c"}, {"out"},
                               std::string(backend::unary_elem_index_src()) + R"(
        float fa = static_cast<float>(a[elem]);
        float fb = static_cast<float>(b[elem]);
        float fc = static_cast<float>(c[elem]);
        float s = fma(fa, fb, fc);
        uint s_bits = as_type<uint>(s);
        if ((s_bits & 0x7f800000u) != 0x7f800000u && (s_bits & 0xffffu) == 0x8000u) {
            float p = fa * fb;
            float err = (metal::abs(p) >= metal::abs(fc)) ? (fma(fa, fb, -s) + fc)
                                                          : fma(fa, fb, fc - s);
            if (err != 0.0f) {
                s_bits = ((err > 0.0f) == (s > 0.0f)) ? (s_bits | 1u) : (s_bits - 1u);
                s = as_type<float>(s_bits);
            }
        }
        out[elem] = static_cast<bfloat>(s);
        )",
                               "", true, false, mx::CompileOptions{});
    const auto launch = backend::metal_launch_1d(n);
    const auto& k = (dt == Dtype::BF16) ? kernel_bf16 : kernel;
    auto outs = k({at_shape(a), at_shape(b), at_shape(c)}, {k_shape}, {mlx_dt}, launch.grid,
                  launch.threadgroup, {}, std::nullopt, false, {});
    return gpu::wrap_mlx_array(
        mlx_shape.empty() ? mx::reshape(outs[0], mlx_shape) : std::move(outs[0]), dt);
}

}  // namespace

TensorImplPtr
fma_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in, const TensorImplPtr& c_in) {
    Validator::input(a_in, "fma.a").non_null();
    Validator::input(b_in, "fma.b").non_null();
    Validator::input(c_in, "fma.c").non_null();

    const Dtype dt = a_in->dtype();
    Validator::pair(a_in, b_in, "fma: a/b").same_dtype();
    Validator::pair(a_in, c_in, "fma: a/c").same_dtype();
    if (!is_floating_point(dt))
        ErrorBuilder("fma").invalid_argument(
            "dtype must be floating-point (F16, BF16, F32, or F64)");

    auto [a1, b1] = detail::align_binary_devices(a_in, b_in, "fma: a/b");
    auto [b, c] = detail::align_binary_devices(b1, c_in, "fma: a/c");
    auto [a, _] = detail::align_binary_devices(a1, b, "fma: a/b");

    const Shape out_shape =
        detail::broadcast_shapes(detail::broadcast_shapes(a->shape(), b->shape()), c->shape());
    const Device dev = a->device();
    OpScopeFull scope{"fma", dev, dt, out_shape};

    Storage out_storage;
    if (PhantomMode::is_enabled()) [[unlikely]] {
        out_storage = phantom_storage(out_shape, dt, dev);
    } else if (dev == Device::GPU && dt != Dtype::F64) {
        out_storage =
            Storage{fma_gpu(std::get<GpuStorage>(a->storage()), std::get<GpuStorage>(b->storage()),
                            std::get<GpuStorage>(c->storage()), out_shape, dt)};
    } else {
        out_storage = to_device_storage(fma_cpu(a, b, c, out_shape, dt), dev, out_shape);
    }

    auto result = fresh(std::move(out_storage), out_shape, dt, dev);
    const std::array<bool, 3> saves{b->requires_grad(), a->requires_grad(), false};
    kernel::NaryKernel<FmaBackward, 3>::wire_autograd(std::make_shared<FmaBackward>(), {a, b, c},
                                                      result, saves);
    return result;
}

const OpSchema FmaBackward::schema_v1{"fma", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> FmaBackward::apply(Storage grad_out) {
    auto g = std::make_shared<TensorImpl>(std::move(grad_out), out_shape_, dtype_, device_, false);
    auto wrap_saved = [this](std::size_t k) {
        return std::make_shared<TensorImpl>(saved_inputs_[k], input_shapes_[k], dtype_, device_,
                                            false);
    };
    Storage da = next_edges_[0].is_valid()
                     ? reduce_grad_to_shape(mul_op(g, wrap_saved(1))->storage(), out_shape_,
                                            input_shapes_[0], dtype_, device_)
                     : Storage{};
    Storage db = next_edges_[1].is_valid()
                     ? reduce_grad_to_shape(mul_op(g, wrap_saved(0))->storage(), out_shape_,
                                            input_shapes_[1], dtype_, device_)
                     : Storage{};
    Storage dc = next_edges_[2].is_valid() ? reduce_grad_to_shape(g->storage(), out_shape_,
                                                                  input_shapes_[2], dtype_, device_)
                                           : Storage{};
    return {std::move(da), std::move(db), std::move(dc)};
}

std::vector<TensorImplPtr> FmaBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    TensorImplPtr da = next_edges_[0].is_valid()
                           ? detail::reduce_impl_to_shape(mul_op(grad_out, saved_impl_inputs_[1]),
                                                          out_shape_, input_shapes_[0])
                           : nullptr;
    TensorImplPtr db = next_edges_[1].is_valid()
                           ? detail::reduce_impl_to_shape(mul_op(grad_out, saved_impl_inputs_[0]),
                                                          out_shape_, input_shapes_[1])
                           : nullptr;
    TensorImplPtr dc = next_edges_[2].is_valid()
                           ? detail::reduce_impl_to_shape(grad_out, out_shape_, input_shapes_[2])
                           : nullptr;
    return {std::move(da), std::move(db), std::move(dc)};
}

LUCID_REGISTER_OP(FmaBackward)

}  // namespace lucid
