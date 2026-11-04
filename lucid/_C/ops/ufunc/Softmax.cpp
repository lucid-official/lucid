// lucid/_C/ops/ufunc/Softmax.cpp
//
// Softmax forward and Jacobian-vector backward.  The backward is fully
// delegated to the backend (IBackend::softmax_backward) which computes:
//   dL/dx_i = p_i * (dL/dy_i - sum_j p_j * dL/dy_j)
// in a single fused pass to avoid materialising the N×N Jacobian.

#include "Softmax.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../bfunc/_BinaryOp.h"
#include "../utils/Promote.h"
#include "../utils/ScalarAxis.h"
#include "Exponential.h"
#include "Reductions.h"

namespace lucid {

// ForceFP32 prevents probability underflow; softmax on float16 is numerically
// unreliable for large logit magnitudes.
const OpSchema SoftmaxBackward::schema_v1{"softmax", 1, AmpPolicy::ForceFP32, true};

// Normalise the axis, dispatch the forward softmax, and wire the backward node.
// The output tensor is saved on the node (saved_output_) so apply() can use p
// directly without re-running the forward pass.
// set_flops(*5): each element needs roughly exp + div + two adds + a compare.
TensorImplPtr SoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "softmax.a").non_null();
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "softmax", softmax_op);

    const int wrapped = wrap_dim(axis, static_cast<int>(a->shape().size()), "softmax");

    // Assembles its own forward, so it asks for the schema dtype the
    // kernel templates would have applied.  Without it an integer input
    // reached the backend untouched: Metal answered [0, 0, 0] for the
    // softmax of [1, 2, 3] — probabilities truncated to integers — while
    // the CPU raised.  See promote_for_schema.
    const TensorImplPtr x = promote_for_schema(schema_v1, a);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    // 3.5 Phase 1.2: report axis so the compile-path Softmax emitter
    // can rebuild the same reduction inside MPSGraph.
    scope.set_attr("dim", static_cast<std::int64_t>(wrapped));
    Storage out_storage = backend::Dispatcher::for_device(x->device())
                              .softmax(x->storage(), x->shape(), wrapped, x->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                               x->device(), false);
    scope.set_flops(static_cast<std::int64_t>(x->numel()) * 5);

    // wire_autograd is called unconditionally so the 3.5 compile-path
    // trace hook fires regardless of GradMode (matches NaryKernel /
    // UnaryKernel / BinaryKernel).  GradMode-disabled and no-requires-grad
    // paths are short-circuited *inside* wire_autograd after the hook.
    auto bwd = std::make_shared<SoftmaxBackward>();
    bwd->keep_output(result->storage());  // p = softmax(x)
    bwd->axis_ = wrapped;
    kernel::NaryKernel<SoftmaxBackward, 1>::wire_autograd(std::move(bwd), {x}, result, false);
    return result;
}

// dL/dx = p * (dL/dy - dot(dL/dy, p)) along axis_, computed by the backend.
std::vector<Storage> SoftmaxBackward::apply(Storage grad_out) {
    return {backend::Dispatcher::for_device(device_).softmax_backward(
        saved_output_, grad_out, input_shapes_[0], axis_, dtype_)};
}

TensorImplPtr softmax_op(const TensorImplPtr& a, int axis) {
    return SoftmaxBackward::forward(a, axis);
}
std::vector<TensorImplPtr> SoftmaxBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // dL/dx = p * (g - sum(g * p))  along axis_, with p recomputed so it
    // carries its own dependence on x.
    const auto& x = saved_impl_inputs_[0];
    TensorImplPtr p;
    if (x) {
        p = softmax_op(x, axis_);
    } else if (::lucid::holds_saved_values(saved_output_)) {
        p = std::make_shared<TensorImpl>(saved_output_, out_shape_, dtype_, device_, true);
        p->set_grad_fn(this->shared_from_this());
        p->set_leaf(false);
    } else {
        ErrorBuilder("softmax").fail("graph-mode backward is missing its saved input");
    }
    auto dot = sum_op(mul_op(grad_out, p), std::vector<int>{axis_}, /*keepdims=*/true);
    return {mul_op(p, sub_op(grad_out, dot))};
}

LUCID_REGISTER_OP(SoftmaxBackward)

// log_softmax — ForceFP32 for the same numerical stability reason as softmax.
const OpSchema LogSoftmaxBackward::schema_v1{"log_softmax", 1, AmpPolicy::ForceFP32, true};

// Forward: compute log_softmax via the numerically-stable backend,
// save the log_softmax output for use in the backward pass.
TensorImplPtr LogSoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "log_softmax.a").non_null();
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "log_softmax", log_softmax_op);

    const int wrapped = wrap_dim(axis, static_cast<int>(a->shape().size()), "log_softmax");

    // Assembles its own forward, so it asks for the schema dtype the
    // kernel templates would have applied.  Without it an integer input
    // reached the backend untouched: Metal answered [0, 0, 0] for the
    // softmax of [1, 2, 3] — probabilities truncated to integers — while
    // the CPU raised.  See promote_for_schema.
    const TensorImplPtr x = promote_for_schema(schema_v1, a);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    // 3.5 Phase 1.2: report axis for the compile-path LogSoftmax emitter.
    scope.set_attr("dim", static_cast<std::int64_t>(wrapped));
    Storage out_storage = backend::Dispatcher::for_device(x->device())
                              .log_softmax(x->storage(), x->shape(), wrapped, x->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                               x->device(), false);

    // Unconditional wire_autograd — see SoftmaxBackward::forward for the
    // rationale (trace-hook visibility under no-grad / no-requires-grad).
    auto bwd = std::make_shared<LogSoftmaxBackward>();
    bwd->keep_output(result->storage());  // y = log_softmax(x)
    bwd->axis_ = wrapped;
    kernel::NaryKernel<LogSoftmaxBackward, 1>::wire_autograd(std::move(bwd), {x}, result, false);
    return result;
}

// Backward: dL/dx = dL/dy - exp(y) * sum(dL/dy, axis)
// where exp(y) = softmax(x) (probabilities).  Delegated to the backend so
// that broadcasting (sum_g expanded back to input shape) is handled correctly.
std::vector<Storage> LogSoftmaxBackward::apply(Storage grad_out) {
    const Shape& shape = input_shapes_[0];
    return {backend::Dispatcher::for_device(device_).log_softmax_backward(saved_output_, grad_out,
                                                                          shape, axis_, dtype_)};
}

TensorImplPtr log_softmax_op(const TensorImplPtr& a, int axis) {
    return LogSoftmaxBackward::forward(a, axis);
}
std::vector<TensorImplPtr> LogSoftmaxBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // dL/dx = g - softmax(x) * sum(g)  along axis_.
    const auto& x = saved_impl_inputs_[0];
    TensorImplPtr p;
    if (x) {
        p = softmax_op(x, axis_);
    } else if (::lucid::holds_saved_values(saved_output_)) {
        auto y = std::make_shared<TensorImpl>(saved_output_, out_shape_, dtype_, device_, true);
        y->set_grad_fn(this->shared_from_this());
        y->set_leaf(false);
        p = exp_op(y);
    } else {
        ErrorBuilder("log_softmax").fail("graph-mode backward is missing its saved input");
    }
    auto total = sum_op(grad_out, std::vector<int>{axis_}, /*keepdims=*/true);
    return {sub_op(grad_out, mul_op(p, total))};
}

LUCID_REGISTER_OP(LogSoftmaxBackward)

}  // namespace lucid
