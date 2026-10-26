// lucid/_C/ops/bfunc/Floordiv.cpp
//
// The floor-division family: floordiv_op, fmod_op and remainder_op and
// their backward nodes.  The element rule is backend/DivMod.h; the CPU and
// Metal kernels are backend/{cpu,gpu}/DivMod.h.

#include "Floordiv.h"

#include <string>
#include <variant>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/DivMod.h"
#include "../../backend/cpu/DivMod.h"
#include "../../backend/gpu/DivMod.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../core/TensorImpl.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Discrete.h"
#include "Div.h"
#include "Mul.h"
#include "_Detail.h"
#include "_Opmath.h"

namespace lucid {

namespace {

using backend::DivMod;

// The 16-bit floats are computed in float32 and rounded once.
CpuStorage
cpu_divmod(DivMod mode, const CpuStorage& a, const CpuStorage& b, const Shape& shape, Dtype dt) {
    const std::size_t n = shape_numel(shape);
    if (!is_half_float(dt))
        return backend::cpu::divmod(mode, a, b, n, dt);
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    const Storage a32 = be.astype(Storage{a}, shape, dt, Dtype::F32);
    const Storage b32 = be.astype(Storage{b}, shape, dt, Dtype::F32);
    const Storage q{backend::cpu::divmod(mode, std::get<CpuStorage>(a32), std::get<CpuStorage>(b32),
                                         n, Dtype::F32)};
    return std::get<CpuStorage>(be.astype(q, shape, Dtype::F32, dt));
}

GpuStorage gpu_divmod(DivMod mode, const GpuStorage& a, const GpuStorage& b, Dtype dt) {
    return gpu::wrap_mlx_array(backend::gpu_divmod(mode, *a.arr, *b.arr), dt);
}

Storage divmod_storage(
    DivMod mode, const Storage& a, const Storage& b, const Shape& shape, Dtype dt, Device device) {
    if (PhantomMode::is_enabled() || storage_is_phantom(a) || storage_is_phantom(b))
        return phantom_storage(shape, dt, device);
    if (device == Device::GPU)
        return Storage{gpu_divmod(mode, std::get<GpuStorage>(a), std::get<GpuStorage>(b), dt)};
    return Storage{cpu_divmod(mode, std::get<CpuStorage>(a), std::get<CpuStorage>(b), shape, dt)};
}

// The checks every op of the family makes before it computes.  A zero
// integer divisor is refused here on the CPU, as the reference's CPU does.
// Metal is not asked: it would have to read ``b`` back, a host stall on
// every integer ``//``; it answers the defined value in backend/DivMod.h.
std::pair<TensorImplPtr, TensorImplPtr>
check_operands(const char* op, const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    auto [a, b] = bfunc_detail::align_pair(a_in, b_in, op);
    const Dtype dt = a->dtype();
    if (!is_integral(dt) && !is_floating_point(dt))
        ErrorBuilder(op).not_implemented(std::string("dtype ") + std::string(dtype_name(dt)) +
                                         " is not supported; " + op +
                                         " takes integer or real floating-point operands");
    if (is_integral(dt) && a->numel() > 0 && b->device() == Device::CPU && !b->is_phantom()) {
        if (backend::cpu::any_zero_integer(std::get<CpuStorage>(b->storage()), b->numel(), dt))
            ErrorBuilder(op).fail("ZeroDivisionError: integer division by zero");
    }
    return {std::move(a), std::move(b)};
}

}  // namespace

// ── floordiv ────────────────────────────────────────────────────────────────

const OpSchema FloordivBackward::schema_v1{"floordiv", 1, AmpPolicy::Promote, true};

CpuStorage FloordivBackward::cpu_kernel(const CpuStorage& a,
                                        const CpuStorage& b,
                                        const Shape& shape,
                                        Dtype dt) {
    return cpu_divmod(DivMod::FloorDiv, a, b, shape, dt);
}

GpuStorage
FloordivBackward::gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape&, Dtype dt) {
    return gpu_divmod(DivMod::FloorDiv, a, b, dt);
}

std::pair<Storage, Storage> FloordivBackward::grad_formula(const Storage&) {
    return {make_zero_storage(out_shape_, dtype_, device_),
            make_zero_storage(out_shape_, dtype_, device_)};
}

std::pair<TensorImplPtr, TensorImplPtr> FloordivBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr&, const TensorImplPtr&) {
    return {zeros_like_op(grad_out), zeros_like_op(grad_out)};
}

TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (auto r = opmath::try_scalar<opmath::Op::FloorDiv>(a, b, &floordiv_op))
        return r;
    const auto [a_dev, b_dev] = check_operands("floordiv", a, b);
    return FloordivBackward::forward(a_dev, b_dev);
}

LUCID_REGISTER_OP(FloordivBackward)

// ── fmod ────────────────────────────────────────────────────────────────────

const OpSchema FmodBackward::schema_v1{"fmod", 1, AmpPolicy::Promote, true};

CpuStorage
FmodBackward::cpu_kernel(const CpuStorage& a, const CpuStorage& b, const Shape& shape, Dtype dt) {
    return cpu_divmod(DivMod::Fmod, a, b, shape, dt);
}

GpuStorage
FmodBackward::gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape&, Dtype dt) {
    return gpu_divmod(DivMod::Fmod, a, b, dt);
}

// ``db = -g * trunc(a / b)``: the quotient the reference differentiates
// with is its ``div(rounding_mode="trunc")``, which for floats is
// ``trunc(a / b)``.
std::pair<Storage, Storage> FmodBackward::grad_formula(const Storage& grad_out) {
    Storage da = needs_input_grad(0) ? grad_out : Storage{};
    if (!needs_input_grad(1))
        return {std::move(da), Storage{}};
    const std::size_t n = shape_numel(out_shape_);
    const Storage a_b = saved_input_broadcasted(0);
    const Storage b_b = saved_input_broadcasted(1);
    auto& be = backend::Dispatcher::for_device(device_);
    const Storage q = TruncBackward::dispatch(be, divide_storages(a_b, b_b, n, dtype_, device_),
                                              out_shape_, dtype_);
    Storage db =
        negate_storage(multiply_storages(grad_out, q, n, dtype_, device_), n, dtype_, device_);
    return {std::move(da), std::move(db)};
}

std::pair<TensorImplPtr, TensorImplPtr> FmodBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    TensorImplPtr da = needs_input_grad(0) ? grad_out : nullptr;
    TensorImplPtr db =
        needs_input_grad(1) ? neg_op(mul_op(grad_out, trunc_op(div_op(a, b)))) : nullptr;
    return {std::move(da), std::move(db)};
}

TensorImplPtr fmod_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    const auto [a_dev, b_dev] = check_operands("fmod", a, b);
    return FmodBackward::forward(a_dev, b_dev);
}

LUCID_REGISTER_OP(FmodBackward)

// ── remainder ───────────────────────────────────────────────────────────────

const OpSchema RemainderBackward::schema_v1{"remainder", 1, AmpPolicy::Promote, true};

CpuStorage RemainderBackward::cpu_kernel(const CpuStorage& a,
                                         const CpuStorage& b,
                                         const Shape& shape,
                                         Dtype dt) {
    return cpu_divmod(DivMod::Remainder, a, b, shape, dt);
}

GpuStorage
RemainderBackward::gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape&, Dtype dt) {
    return gpu_divmod(DivMod::Remainder, a, b, dt);
}

// ``db = -g * (a // b)``, with the exact floored quotient.
std::pair<Storage, Storage> RemainderBackward::grad_formula(const Storage& grad_out) {
    Storage da = needs_input_grad(0) ? grad_out : Storage{};
    if (!needs_input_grad(1))
        return {std::move(da), Storage{}};
    const std::size_t n = shape_numel(out_shape_);
    const Storage q = divmod_storage(DivMod::FloorDiv, saved_input_broadcasted(0),
                                     saved_input_broadcasted(1), out_shape_, dtype_, device_);
    Storage db =
        negate_storage(multiply_storages(grad_out, q, n, dtype_, device_), n, dtype_, device_);
    return {std::move(da), std::move(db)};
}

std::pair<TensorImplPtr, TensorImplPtr> RemainderBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    TensorImplPtr da = needs_input_grad(0) ? grad_out : nullptr;
    TensorImplPtr db = needs_input_grad(1) ? neg_op(mul_op(grad_out, floordiv_op(a, b))) : nullptr;
    return {std::move(da), std::move(db)};
}

TensorImplPtr remainder_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    const auto [a_dev, b_dev] = check_operands("remainder", a, b);
    return RemainderBackward::forward(a_dev, b_dev);
}

LUCID_REGISTER_OP(RemainderBackward)

}  // namespace lucid
