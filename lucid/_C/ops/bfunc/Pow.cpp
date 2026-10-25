// lucid/_C/ops/bfunc/Pow.cpp
//
// Implements PowBackward::grad_formula and the pow_op free function.

#include "Pow.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../complex/Imag.h"
#include "../complex/Real.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Exponential.h"
#include "../utils/Select.h"
#include "Compare.h"
#include "Mul.h"
#include "Sub.h"

namespace lucid {

// ForceFP32: under autocast a floating power runs in float32, as the
// reference runs it — log() and fractional exponents are unstable in half
// precision, and ``x ** 2.0`` (pow_scalar) already answered float32.
// ``integral_closed``: an integer or bool power is exact in its own dtype,
// inside autocast and out, so the policy's int-to-float32 promotion is off.
const OpSchema PowBackward::schema_v1{
    "pow", 1, AmpPolicy::ForceFP32, true, "", /*real=*/false, /*closed=*/true};

// Gradient of element-wise power (a ^ b).
//
// Power rule (gradient w.r.t. a):
//   dA = where(b == 0, 0, b * a^(b-1)) * grad_out
//
// Exponential rule (gradient w.r.t. b):
//   dB = where(a == 0 && b >= 0, 0, log(a) * a^b) * grad_out
//
// Both inputs are retrieved at the broadcast output shape before any arithmetic.
std::pair<Storage, Storage> PowBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const bool need_a = needs_input_grad(0);
    const bool need_b = needs_input_grad(1);
    if (!need_a && !need_b)
        return {Storage{}, Storage{}};

    Storage a_buf = saved_input_broadcasted(0);
    Storage b_buf = saved_input_broadcasted(1);
    const auto& a = a_buf;
    const auto& b = b_buf;
    auto& be = backend::Dispatcher::for_device(device_);
    Storage zero = make_zero_storage(out_shape_, dtype_, device_);

    // dA = where(b == 0, 0, b * a^(b-1)) * grad_out
    Storage dx;
    if (need_a) {
        Storage b_minus_one = add_scalar_storage(b, -1.0, n, dtype_, device_);
        Storage a_pow_bm1 = pow_storage(a, b_minus_one, n, dtype_, device_);
        Storage b_times = multiply_storages(b, a_pow_bm1, n, dtype_, device_);
        Storage b_eq_zero = be.compare_binary(b, zero, out_shape_, dtype_, 0);
        b_times = be.where_op(b_eq_zero, zero, b_times, out_shape_, dtype_);
        dx = multiply_storages(b_times, grad_out, n, dtype_, device_);
    }

    // dB = where(a == 0 && b >= 0, 0, log(a) * a^b) * grad_out
    Storage dy;
    if (need_b) {
        Storage log_a = log_storage(a, n, dtype_, device_);
        Storage a_pow_b = pow_storage(a, b, n, dtype_, device_);
        Storage prod = multiply_storages(log_a, a_pow_b, n, dtype_, device_);
        Storage a_eq_zero = be.compare_binary(a, zero, out_shape_, dtype_, 0);
        if (is_complex(dtype_)) {
            const Dtype real_dt = dtype_ == Dtype::C128 ? Dtype::F64 : Dtype::F32;
            Storage zero_r = make_zero_storage(out_shape_, real_dt, device_);
            Storage re_ge =
                be.compare_binary(be.complex_real(b, out_shape_), zero_r, out_shape_, real_dt, 3);
            Storage im_eq =
                be.compare_binary(be.complex_imag(b, out_shape_), zero_r, out_shape_, real_dt, 0);
            Storage im_masked = be.where_op(im_eq, zero, prod, out_shape_, dtype_);
            Storage b_masked = be.where_op(re_ge, im_masked, prod, out_shape_, dtype_);
            prod = be.where_op(a_eq_zero, b_masked, prod, out_shape_, dtype_);
        } else {
            Storage b_ge = be.compare_binary(b, zero, out_shape_, dtype_, 3);
            Storage b_masked = be.where_op(b_ge, zero, prod, out_shape_, dtype_);
            prod = be.where_op(a_eq_zero, b_masked, prod, out_shape_, dtype_);
        }
        dy = multiply_storages(prod, grad_out, n, dtype_, device_);
    }

    return {std::move(dx), std::move(dy)};
}

std::pair<TensorImplPtr, TensorImplPtr> PowBackward::grad_formula_impl(const TensorImplPtr& g,
                                                                       const TensorImplPtr& a_ptr,
                                                                       const TensorImplPtr& b_ptr) {
    auto ones = ones_like_op(a_ptr, false);
    auto zero_a = zeros_like_op(a_ptr, false);
    auto zero_b = zeros_like_op(b_ptr, false);

    // dA = where(b == 0, 0, b * a^(b-1)) * g
    TensorImplPtr bm1 = sub_op(b_ptr, ones);
    TensorImplPtr a_pow_bm1 = pow_op(a_ptr, bm1);
    TensorImplPtr b_times = where_op(equal_op(b_ptr, zero_b), zero_a, mul_op(b_ptr, a_pow_bm1));
    TensorImplPtr dA = mul_op(b_times, g);

    // dB = where(a == 0 && b >= 0, 0, log(a) * a^b) * g
    TensorImplPtr log_a = log_op(a_ptr);
    TensorImplPtr a_pow_b = pow_op(a_ptr, b_ptr);
    TensorImplPtr prod = mul_op(log_a, a_pow_b);
    TensorImplPtr a_eq_zero = equal_op(a_ptr, zero_a);
    if (is_complex(b_ptr->dtype())) {
        auto b_re = real_op(b_ptr);
        auto b_im = imag_op(b_ptr);
        auto zero_r = zeros_like_op(b_re, false);
        auto im_zero = where_op(equal_op(b_im, zero_r), zero_b, prod);
        auto b_masked = where_op(greater_equal_op(b_re, zero_r), im_zero, prod);
        prod = where_op(a_eq_zero, b_masked, prod);
    } else {
        auto b_masked = where_op(greater_equal_op(b_ptr, zero_b), zero_b, prod);
        prod = where_op(a_eq_zero, b_masked, prod);
    }
    TensorImplPtr dB = mul_op(prod, g);

    return {std::move(dA), std::move(dB)};
}

TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return PowBackward::forward(a, b);
}

LUCID_REGISTER_OP(PowBackward)

}  // namespace lucid
