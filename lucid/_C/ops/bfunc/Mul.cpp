// lucid/_C/ops/bfunc/Mul.cpp
//
// Implements MulBackward::grad_formula and the mul_op free function.

#include "Mul.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "_Opmath.h"

namespace lucid {

const OpSchema MulBackward::schema_v1{"mul", 1, AmpPolicy::Promote, true};

// Gradient of element-wise multiplication (product rule).
//
// saved_input_broadcasted(k) returns saved_inputs_[k] already expanded to
// out_shape_ if the original input was narrower due to broadcasting.  This
// expansion is required so that element-wise multiplication with grad_out (also
// at out_shape_) is shape-compatible.
//
// dA = grad_out ⊙ b_broadcast
// dB = grad_out ⊙ a_broadcast
//
// BinaryKernel::apply will subsequently reduce both results back to the
// original input shapes via reduce_grad_to_shape, and conjugates around this
// formula on a complex dtype (grad_out ⊙ conj(b)).
std::pair<Storage, Storage> MulBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const bool need_a = needs_input_grad(0);
    const bool need_b = needs_input_grad(1);

    auto factor = [&](std::size_t k) -> Storage {
        if (device_ == Device::CPU && shape_numel(input_shapes_[k]) == 1 && n > 1 &&
            std::holds_alternative<CpuStorage>(saved_inputs_[k]) &&
            std::holds_alternative<CpuStorage>(grad_out)) {
            const auto& cs = std::get<CpuStorage>(saved_inputs_[k]);
            if (auto s = detail::read_cpu_scalar(cs, dtype_))
                return mul_scalar_storage(grad_out, *s, n, dtype_, device_);
        }

        return multiply_storages(grad_out, saved_input_broadcasted(k), n, dtype_, device_);
    };

    return {
        need_a ? factor(1) : Storage{},
        need_b ? factor(0) : Storage{},
    };
}

std::pair<TensorImplPtr, TensorImplPtr> MulBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    // da = grad_out * b,  db = grad_out * a.
    // a and b are already broadcast-expanded to out_shape_ by BinaryKernel.
    TensorImplPtr da = needs_input_grad(0) ? mul_op(grad_out, b) : nullptr;
    TensorImplPtr db = needs_input_grad(1) ? mul_op(grad_out, a) : nullptr;
    return {std::move(da), std::move(db)};
}

TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (auto r = opmath::try_scalar<opmath::Op::Mul>(a, b, &mul_op))
        return r;
    return MulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MulBackward)

}  // namespace lucid
