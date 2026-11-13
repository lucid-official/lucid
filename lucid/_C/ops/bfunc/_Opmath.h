// lucid/_C/ops/bfunc/_Opmath.h
//
// Op-math for a half tensor and a wider 0-d operand.
//
// ``mul_op``, ``div_op`` and ``floordiv_op`` accept one mixed pair that
// BinaryKernel refuses: a float16 / bfloat16 tensor ``T`` of any shape and a
// 0-d float32 operand ``S`` (float64 too on the CPU stream, which is the only
// stream that holds it), in either order.  ``S`` is read at its own width,
// the op runs in float32 and the result is rounded once into ``T``'s dtype.
// Casting ``S`` to the half dtype first — all a same-dtype pair can do — is
// what made ``ones(f16) / 1e5`` zero (1e-5 underflows float16),
// ``full(1e-3, f16) * 1e5`` inf (1e5 overflows it) and ``bf16(3.09375) * 0.1``
// 0.3105 instead of 0.3086.  Only these three ops: the reference applies
// op-math to a scalar operand of mul, div and floor_divide and rounds it first
// for every other binary op, which Lucid already does.
//
// Floor division is the exact float32 floor quotient, rounded once — the
// same rule as every other floating-point ``//`` in Lucid (backend/DivMod.h).
//
// When the 0-d operand requires grad, its gradient is reduced over ``T`` in
// float32 (and cast back to float64 when ``S`` is float64 on the CPU) so a
// learnable 0-d float32 parameter keeps op-math in forward and backward
// (LCD-300).  Under autocast, when the policy moves the half dtype elsewhere,
// both operands are cast to the autocast dtype and the ordinary same-dtype op
// runs.
//
// The Metal stream fuses widen, op and narrow into one kernel with the 0-d
// operand as a kernel input.  The old route through a same-dtype pair spent
// a pass materialising the broadcast 0-d before the multiply.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/DivMod.h"
#include "../../backend/cpu/DivMod.h"
#include "../../backend/gpu/DivMod.h"
#include "../../backend/gpu/HalfAccumulation.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Device.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpSchema.h"
#include "../../core/PhantomMode.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Reductions.h"
#include "../ufunc/ToDevice.h"
#include "Div.h"
#include "Mul.h"

namespace lucid::opmath {

enum class Op : std::uint8_t { Mul, Div, FloorDiv };

constexpr std::string_view op_name(Op op) {
    switch (op) {
    case Op::Mul:
        return "mul";
    case Op::Div:
        return "div";
    case Op::FloorDiv:
        return "floordiv";
    }
    return "mul";
}

// The schema whose AMP policy decides the op's effective dtype.  Floor
// division of floats is ``floor(div)``, so it follows ``div``.
inline const OpSchema& amp_schema(Op op) {
    return op == Op::Mul ? MulBackward::schema_v1 : DivBackward::schema_v1;
}

struct Pair {
    TensorImplPtr t;
    TensorImplPtr s;
    bool scalar_first;
};

inline bool is_wide_scalar(const TensorImplPtr& s) {
    return s->shape().empty() &&
           (s->dtype() == Dtype::F32 || (s->dtype() == Dtype::F64 && s->device() == Device::CPU));
}

// ``(a, b)`` as an op-math pair, or nothing — every other mixed pair is left
// to BinaryKernel, which rejects it as before.
inline std::optional<Pair> match(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b || a->dtype() == b->dtype())
        return std::nullopt;
    auto dev_ok = [](const TensorImplPtr& t, const TensorImplPtr& s) {
        return t->device() == s->device() ||
               (t->device() == Device::GPU && s->device() == Device::CPU);
    };
    if (is_half_float(a->dtype()) && is_wide_scalar(b) && dev_ok(a, b))
        return Pair{a, b, false};
    if (is_half_float(b->dtype()) && is_wide_scalar(a) && dev_ok(b, a))
        return Pair{b, a, true};
    return std::nullopt;
}

namespace gpu_fn {

namespace mx = ::mlx::core;

// ``in = {T, S}``; the result in ``T``'s dtype.  One type per (op, order):
// ``mlx_fused`` keeps one compiled kernel per functor type.
template <Op K, bool ScalarFirst>
struct Forward {
    std::vector<mx::array> operator()(const std::vector<mx::array>& in) const {
        const mx::array t = backend::widen_half(in[0]);
        const mx::array& s = in[1];
        mx::array r = s;
        if constexpr (K == Op::Mul)
            r = mx::multiply(t, s);
        else if constexpr (K == Op::Div)
            r = ScalarFirst ? mx::divide(s, t) : mx::divide(t, s);
        else
            r = ScalarFirst ? backend::gpu_floor_divide_float32(s, t)
                            : backend::gpu_floor_divide_float32(t, s);
        return {backend::narrow_to(r, in[0].dtype())};
    }
};

// ``in = {g, T, S}``: the gradient of ``S / T`` with respect to ``T``,
// ``-(g * (S / T)) / T``, in float32 and rounded once into ``g``'s dtype.  ``T^2`` is
// never formed: it leaves float32 for a bfloat16 ``T`` outside ~1e-19 .. 1e19.
struct ReciprocalGrad {
    std::vector<mx::array> operator()(const std::vector<mx::array>& in) const {
        const mx::array t = backend::widen_half(in[1]);
        const mx::array r = mx::negative(
            mx::divide(mx::multiply(backend::widen_half(in[0]), mx::divide(in[2], t)), t));
        return {backend::narrow_to(r, in[0].dtype())};
    }
};

inline mx::array fused(Op op, bool scalar_first, const std::vector<mx::array>& ins) {
    switch (op) {
    case Op::Mul:
        return backend::mlx_fused(ins, Forward<Op::Mul, false>{})[0];
    case Op::Div:
        return scalar_first ? backend::mlx_fused(ins, Forward<Op::Div, true>{})[0]
                            : backend::mlx_fused(ins, Forward<Op::Div, false>{})[0];
    case Op::FloorDiv:
        // Unfused: a fused kernel is built for fast math, which loses the
        // infinities floor division answers by (backend/gpu/DivMod.h).
        return scalar_first ? Forward<Op::FloorDiv, true>{}(ins)[0]
                            : Forward<Op::FloorDiv, false>{}(ins)[0];
    }
    return backend::mlx_fused(ins, Forward<Op::Mul, false>{})[0];
}

inline Storage wrap(mx::array r, Dtype dt) {
    // A strided ``T`` gives a strided result; everything downstream of a
    // Storage assumes row-major bytes (engine-mlx-data-ignores-strides).
    return Storage{gpu::wrap_mlx_array(mx::contiguous(r), dt)};
}

inline const mx::array& arr(const Storage& s) {
    return *std::get<GpuStorage>(s).arr;
}

}  // namespace gpu_fn

inline double read_cpu_scalar(const Storage& s, Dtype dt) {
    const auto& c = std::get<CpuStorage>(s);
    return dt == Dtype::F64 ? *reinterpret_cast<const double*>(c.ptr.get())
                            : static_cast<double>(*reinterpret_cast<const float*>(c.ptr.get()));
}

// ``x op s`` (or ``s op x``) for ``x`` of dtype ``h`` and shape ``shape`` and
// a 0-d ``s`` of dtype ``s_dt``: computed in float32, rounded once to ``h``.
inline Storage compute(Op op,
                       bool scalar_first,
                       const Storage& x,
                       const Shape& shape,
                       Dtype h,
                       const Storage& s,
                       Dtype s_dt,
                       Device device) {
    if (PhantomMode::is_enabled() || storage_is_phantom(x) || storage_is_phantom(s))
        return phantom_storage(shape, h, device);
    if (device == Device::GPU) {
        return gpu_fn::wrap(gpu_fn::fused(op, scalar_first, {gpu_fn::arr(x), gpu_fn::arr(s)}), h);
    }
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    // The half ``mul_scalar`` already widens, runs vsmul with float(s) and
    // rounds once.
    if (op == Op::Mul)
        return be.mul_scalar(x, shape, h, read_cpu_scalar(s, s_dt));
    // Accelerate has a vector-by-vector division and the backend nothing
    // narrower, so the 0-d is spread to ``shape`` first.  Each lane is still
    // one IEEE float32 division of float(T) by float(S).
    const Storage x32 = be.astype(x, shape, h, Dtype::F32);
    const Storage s32 = s_dt == Dtype::F32 ? s : be.astype(s, Shape{}, s_dt, Dtype::F32);
    const Storage s_full{
        detail::broadcast_cpu(std::get<CpuStorage>(s32), Shape{}, shape, Dtype::F32)};
    const Storage& num = scalar_first ? s_full : x32;
    const Storage& den = scalar_first ? x32 : s_full;
    const Storage q = op == Op::FloorDiv
                          ? Storage{backend::cpu::divmod(
                                backend::DivMod::FloorDiv, std::get<CpuStorage>(num),
                                std::get<CpuStorage>(den), shape_numel(shape), Dtype::F32)}
                          : be.div(num, den, shape, Dtype::F32);
    return be.astype(q, shape, Dtype::F32, h);
}

// ``-(g * (S / T)) / T`` in float32, rounded once to ``h``.  The textbook
// ``-(S * g) / T^2`` forms ``T^2``, which leaves float32 for a bfloat16 ``T`` outside
// ~1e-19 .. 1e19 while the gradient is an ordinary float (LCD-998).
inline Storage reciprocal_grad(const Storage& g,
                               const Storage& t,
                               const Shape& shape,
                               Dtype h,
                               const Storage& s,
                               Dtype s_dt,
                               Device device) {
    if (PhantomMode::is_enabled() || storage_is_phantom(g) || storage_is_phantom(t) ||
        storage_is_phantom(s))
        return phantom_storage(shape, h, device);
    if (device == Device::GPU) {
        return gpu_fn::wrap(backend::mlx_fused({gpu_fn::arr(g), gpu_fn::arr(t), gpu_fn::arr(s)},
                                               gpu_fn::ReciprocalGrad{})[0],
                            h);
    }
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    const Storage g32 = be.astype(g, shape, h, Dtype::F32);
    const Storage t32 = be.astype(t, shape, h, Dtype::F32);
    // Accelerate has no scalar-by-vector division: the 0-d is spread to ``shape`` first.
    const Storage s32 = s_dt == Dtype::F32 ? s : be.astype(s, Shape{}, s_dt, Dtype::F32);
    const Storage s_full{
        detail::broadcast_cpu(std::get<CpuStorage>(s32), Shape{}, shape, Dtype::F32)};
    const Storage s_over_t = be.div(s_full, t32, shape, Dtype::F32);
    const Storage g_s_over_t = be.mul(g32, s_over_t, shape, Dtype::F32);
    const Storage quotient = be.div(g_s_over_t, t32, shape, Dtype::F32);
    return be.astype(be.neg(quotient, shape, Dtype::F32), shape, Dtype::F32, h);
}

// One node per op.  Slot ``t_index_`` is ``T``; slot ``1 - t_index_`` is the
// 0-d operand ``S``, whose gradient is reduced in float32 (and cast back to
// ``s_dtype_`` when ``S`` is float64 on the CPU).
template <Op K>
class ScalarBackward : public kernel::NaryKernel<ScalarBackward<K>, 2> {
public:
    static inline const OpSchema schema_v1{op_name(K), 1, AmpPolicy::Promote, true};

    std::size_t t_index_ = 0;
    Dtype s_dtype_ = Dtype::F32;

    std::string node_name() const override { return std::string(schema_v1.name); }

    std::vector<Storage> apply(Storage grad_out) override {
        const auto& edges = this->next_edges();
        const std::size_t s_index = 1 - t_index_;
        std::vector<Storage> grads(2, Storage{CpuStorage{}});
        if (edges.size() > t_index_ && edges[t_index_].is_valid())
            grads[t_index_] = grad_t(grad_out);
        if (edges.size() > s_index && edges[s_index].is_valid())
            grads[s_index] = grad_s(grad_out);
        return grads;
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const auto& edges = this->next_edges();
        const std::size_t s_index = 1 - t_index_;
        const TensorImplPtr& t = this->saved_impl_inputs_[t_index_];
        const TensorImplPtr& s = this->saved_impl_inputs_[s_index];
        std::vector<TensorImplPtr> grads(2);
        if (edges.size() > t_index_ && edges[t_index_].is_valid())
            grads[t_index_] = graph_grad_t(grad_out, t, s);
        if (edges.size() > s_index && edges[s_index].is_valid())
            grads[s_index] = graph_grad_s(grad_out, t, s);
        return grads;
    }

private:
    bool scalar_first() const { return t_index_ == 1; }

    TensorImplPtr graph_grad_t(const TensorImplPtr& grad_out,
                               const TensorImplPtr& t,
                               const TensorImplPtr& s) const {
        if constexpr (K == Op::Mul) {
            return mul_op(grad_out, s);
        } else if constexpr (K == Op::Div) {
            if (!scalar_first())
                return div_op(grad_out, s);
            const TensorImplPtr g32 = astype_op(grad_out, Dtype::F32);
            const TensorImplPtr t32 = astype_op(t, Dtype::F32);
            const TensorImplPtr s32 = s->dtype() == Dtype::F32 ? s : astype_op(s, Dtype::F32);
            // -(g * (S / T)) / T: T^2 is never formed (see reciprocal_grad).
            return astype_op(neg_op(div_op(mul_op(g32, div_op(s32, t32)), t32)), this->dtype_);
        } else {
            return zeros_like_op(t);
        }
    }

    TensorImplPtr graph_grad_s(const TensorImplPtr& grad_out,
                               const TensorImplPtr& t,
                               const TensorImplPtr& s) const {
        if constexpr (K == Op::FloorDiv) {
            return zeros_like_op(s);
        } else {
            const TensorImplPtr g32 = astype_op(grad_out, Dtype::F32);
            const TensorImplPtr t32 = astype_op(t, Dtype::F32);
            TensorImplPtr term;
            if constexpr (K == Op::Mul) {
                term = mul_op(g32, t32);
            } else {
                if (scalar_first()) {
                    term = div_op(g32, t32);
                } else {
                    const TensorImplPtr s32 =
                        s->dtype() == Dtype::F32 ? s : astype_op(s, Dtype::F32);
                    // -(g * (t / s)) / s: s^2 leaves float32 for a divisor outside
                    // ~1e-19 .. 1e19 (see reciprocal_grad).
                    term = neg_op(div_op(mul_op(g32, div_op(t32, s32)), s32));
                }
            }
            const TensorImplPtr ds = sum_op(term, {}, false);
            return s->dtype() == Dtype::F32 ? ds : astype_op(ds, s->dtype());
        }
    }

    Storage grad_t(const Storage& g) const {
        const Storage& s = this->saved_inputs_[1 - t_index_];
        if constexpr (K == Op::Mul) {
            return compute(Op::Mul, false, g, this->out_shape_, this->dtype_, s, s_dtype_,
                           this->device_);
        } else if constexpr (K == Op::Div) {
            if (scalar_first())
                return reciprocal_grad(g, this->saved_inputs_[t_index_], this->out_shape_,
                                       this->dtype_, s, s_dtype_, this->device_);
            return compute(Op::Div, false, g, this->out_shape_, this->dtype_, s, s_dtype_,
                           this->device_);
        } else {
            return make_zero_storage(this->out_shape_, this->dtype_, this->device_);
        }
    }

    Storage grad_s(const Storage& g) const {
        if constexpr (K == Op::FloorDiv) {
            return make_zero_storage(Shape{}, s_dtype_, this->device_);
        } else {
            if (PhantomMode::is_enabled() || storage_is_phantom(g))
                return phantom_storage(Shape{}, s_dtype_, this->device_);
            const Storage& t = this->saved_inputs_[t_index_];
            if (this->device_ == Device::GPU) {
                namespace mx = ::mlx::core;
                const mx::array g32 = backend::widen_half(gpu_fn::arr(g));
                const mx::array t32 = backend::widen_half(gpu_fn::arr(t));
                mx::array term = g32;
                if constexpr (K == Op::Mul) {
                    term = mx::multiply(g32, t32);
                } else {
                    if (scalar_first()) {
                        term = mx::divide(g32, t32);
                    } else {
                        const mx::array& s32 = gpu_fn::arr(this->saved_inputs_[1 - t_index_]);
                        term =
                            mx::negative(mx::divide(mx::multiply(g32, mx::divide(t32, s32)), s32));
                    }
                }
                return gpu_fn::wrap(mx::sum(term), s_dtype_);
            }
            const std::size_t n = shape_numel(this->out_shape_);
            if (n == 0)
                return make_zero_storage(Shape{}, s_dtype_, Device::CPU);
            auto& be = backend::Dispatcher::for_device(Device::CPU);
            const Shape flat{static_cast<std::int64_t>(n)};
            const Storage g32 = be.astype(g, flat, this->dtype_, Dtype::F32);
            const Storage t32 = be.astype(t, flat, this->dtype_, Dtype::F32);
            Storage term{CpuStorage{}};
            if constexpr (K == Op::Mul) {
                term = be.mul(g32, t32, flat, Dtype::F32);
            } else {
                if (scalar_first()) {
                    term = be.div(g32, t32, flat, Dtype::F32);
                } else {
                    const Storage& s = this->saved_inputs_[1 - t_index_];
                    const Storage s32 =
                        s_dtype_ == Dtype::F32 ? s : be.astype(s, Shape{}, s_dtype_, Dtype::F32);
                    const Storage s_full{detail::broadcast_cpu(std::get<CpuStorage>(s32), Shape{},
                                                               flat, Dtype::F32)};
                    const Storage t_over_s = be.div(t32, s_full, flat, Dtype::F32);
                    const Storage g_t_over_s = be.mul(g32, t_over_s, flat, Dtype::F32);
                    term = be.neg(be.div(g_t_over_s, s_full, flat, Dtype::F32), flat, Dtype::F32);
                }
            }
            Storage sum32 = be.reduce_sum(term, flat, {{0}, false}, Dtype::F32);
            return s_dtype_ == Dtype::F32 ? sum32 : be.astype(sum32, Shape{}, Dtype::F32, s_dtype_);
        }
    }
};

template <Op K>
TensorImplPtr forward(const Pair& p, const TensorImplPtr& a, const TensorImplPtr& b) {
    const Dtype h = p.t->dtype();
    const Device device = p.t->device();
    const Shape& shape = p.t->shape();
    OpScopeFull scope{op_name(K), device, h, shape};
    Storage out_storage =
        compute(K, p.scalar_first, p.t->storage(), shape, h, p.s->storage(), p.s->dtype(), device);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), shape, h, device, false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    auto bwd = std::make_shared<ScalarBackward<K>>();
    bwd->t_index_ = p.scalar_first ? 1 : 0;
    bwd->s_dtype_ = p.s->dtype();
    ScalarBackward<K>* node = bwd.get();
    // Only what a needed gradient reads is saved: ``grad_t`` reads ``s`` for a
    // product or quotient (and ``t`` too for ``s / t``), while ``grad_s`` reads
    // ``t`` (and ``s`` too for ``t / s``).  A write into ``t`` after ``t * s``
    // when ``s`` does not require grad is then not refused, while a write into
    // ``t`` when ``s`` requires grad is refused, matching the reference.
    const std::size_t s_idx = p.scalar_first ? 0 : 1;
    const std::size_t t_idx = 1 - s_idx;
    std::array<bool, 2> saves{};
    saves[s_idx] = (K != Op::FloorDiv && p.t->requires_grad()) ||
                   (K == Op::Div && !p.scalar_first && p.s->requires_grad());
    saves[t_idx] = (K == Op::Div && p.scalar_first && p.t->requires_grad()) ||
                   (K != Op::FloorDiv && p.s->requires_grad());
    // wire_autograd takes the node's dtype from the first input, which is
    // the 0-d when it leads; the gradient lives in the half dtype.
    if (kernel::NaryKernel<ScalarBackward<K>, 2>::wire_autograd(std::move(bwd), {a, b}, out, saves))
        node->dtype_ = h;
    return out;
}

using SameDtypeOp = TensorImplPtr (*)(const TensorImplPtr&, const TensorImplPtr&);

// The op-math result for an op-math pair, or null for any other pair.
// ``same_dtype`` is the public op, reused when autocast moves the dtype.
template <Op K>
TensorImplPtr try_scalar(const TensorImplPtr& a, const TensorImplPtr& b, SameDtypeOp same_dtype) {
    std::optional<Pair> p = match(a, b);
    if (!p)
        return nullptr;
    TensorImplPtr a_use = a;
    TensorImplPtr b_use = b;
    if (p->t->device() == Device::GPU && p->s->device() == Device::CPU) {
        TensorImplPtr s_gpu = p->s->dtype() == Dtype::F64 ? astype_op(p->s, Dtype::F32) : p->s;
        s_gpu = to_device_op(s_gpu, Device::GPU);
        p->s = s_gpu;
        if (p->scalar_first)
            a_use = s_gpu;
        else
            b_use = s_gpu;
    }
    const SchemaGuard sg{amp_schema(K), p->t->dtype(), p->t->device()};
    const Dtype eff = sg.effective_dtype();
    if (eff != p->t->dtype())
        return same_dtype(astype_op(a_use, eff), astype_op(b_use, eff));
    return forward<K>(*p, a_use, b_use);
}

}  // namespace lucid::opmath
