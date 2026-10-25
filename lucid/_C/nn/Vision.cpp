// lucid/_C/nn/Vision.cpp
//
// Implementations of one_hot, rotate, and the bilinear layer.
//
// one_hot: IBackend::one_hot_forward builds a float (or other dtype) matrix
//   of shape (*input.shape, num_classes) with 1 at the class index and 0
//   elsewhere.  No backward node.
//
// rotate: IBackend::rotate_forward applies a 2-D rotation matrix with center
//   (cx, cy) per image channel.  The angle is negated before passing to the
//   backend so that positive angle_deg produces a counter-clockwise rotation
//   in standard image coordinates.  No backward node.
//
// BilinearLayer: IBackend::bilinear_layer_forward computes
//   y = x1 @ W[d, :, :] @ x2^T + b[d] for each output dimension d.
//   Backward is skipped when no input requires a gradient.
//   The has_bias flag is inferred from input_shapes_[3].empty() in apply().

#include "Vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/View.h"

namespace lucid {

TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes, Dtype out_dtype) {
    Validator::input(input, "one_hot.input").non_null();
    if (num_classes <= 0)
        ErrorBuilder("one_hot").invalid_argument("num_classes must be > 0");
    Shape out_shape = input->shape();
    out_shape.push_back(num_classes);
    OpScopeFull scope{"one_hot", input->device(), out_dtype, out_shape};
    scope.set_attr("num_classes", static_cast<std::int64_t>(num_classes));
    scope.set_attr("out_dtype", static_cast<std::int64_t>(out_dtype));

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.one_hot_forward(input->storage(), input->shape(), num_classes, out_dtype);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, out_dtype,
                                            input->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({input}, out);
    }
    return out;
}

TensorImplPtr rotate_op(const TensorImplPtr& input, double angle_deg, double cy, double cx) {
    Validator::input(input, "rotate.input").non_null();
    require_rank(input->shape(), 4, "rotate: input", "(N, C, H, W)");
    OpScopeFull scope{"rotate", input->device(), input->dtype(), input->shape()};
    scope.set_attr("angle_deg", angle_deg);
    scope.set_attr("cy", cy);
    scope.set_attr("cx", cx);
    const double angle_rad_neg = -angle_deg * (M_PI / 180.0);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.rotate_forward(input->storage(), input->shape(), angle_rad_neg, cx, cy, input->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), input->shape(), input->dtype(),
                                            input->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({input}, out);
    }
    return out;
}

const OpSchema BilinearLayerBackward::schema_v1{"bilinear_layer", 1, AmpPolicy::Promote, true};

namespace {

// Decomposed bilinear-layer dimensions derived from x1, x2, and weight shapes.
struct BilinearShape {
    std::size_t B;     // Product of all leading "batch" dims.
    std::size_t D1;    // Last dim of x1 (== weight.shape[1]).
    std::size_t D2;    // Last dim of x2 (== weight.shape[2]).
    std::size_t Dout;  // Output features (== weight.shape[0]).
};

// Validate that the last dims of s1/s2 match the weight's inner dimensions and
// that all leading batch dims of s1 and s2 are equal.  Returns the flattened
// BilinearShape.  Throws ShapeMismatch on any mismatch.
BilinearShape flatten_bilinear(const Shape& s1, const Shape& s2, const Shape& sw) {
    require_min_rank(s1, 1, "bilinear: x1", "(*, D1)");
    require_min_rank(s2, 1, "bilinear: x2", "(*, D2)");
    require_rank(sw, 3, "bilinear: weight", "(D_out, D1, D2)");
    if (s1.size() != s2.size()) {
        throw ShapeMismatch(s1, s2, "bilinear: x1/x2 must have same rank");
    }
    BilinearShape r;
    r.Dout = static_cast<std::size_t>(sw[0]);
    r.D1 = static_cast<std::size_t>(sw[1]);
    r.D2 = static_cast<std::size_t>(sw[2]);
    if (static_cast<std::size_t>(s1.back()) != r.D1 ||
        static_cast<std::size_t>(s2.back()) != r.D2) {
        throw ShapeMismatch(s1, sw, "bilinear: last dims must match weight");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 1 < s1.size(); ++i) {
        if (s1[i] != s2[i])
            throw ShapeMismatch(s1, s2, "bilinear: batch dims must match");
        b *= static_cast<std::size_t>(s1[i]);
    }
    r.B = b;
    return r;
}

// The kernels read every operand, the bias included, in x1's dtype and on
// x1's device.
void require_bilinear_operands(const TensorImplPtr& x1,
                               const TensorImplPtr& x2,
                               const TensorImplPtr& weight,
                               const TensorImplPtr& bias) {
    Validator::pair(x1, x2, "bilinear_layer: x1/x2").same_device().same_dtype();
    Validator::pair(x1, weight, "bilinear_layer: x1/weight").same_device().same_dtype();
    if (bias)
        Validator::pair(x1, bias, "bilinear_layer: x1/bias").same_device().same_dtype();
}

}  // namespace

TensorImplPtr BilinearLayerBackward::forward(const TensorImplPtr& x1_in,
                                             const TensorImplPtr& x2_in,
                                             const TensorImplPtr& weight_in,
                                             const TensorImplPtr& bias_in) {
    if (!x1_in || !x2_in || !weight_in)
        ErrorBuilder("bilinear_layer").fail("null input");
    // The row is WIDEST under Metal autocast: all four operands meet in one
    // dtype, as the reference's do, rather than failing the check below.
    const auto operands = promote_for_schema(schema_v1, {x1_in, x2_in, weight_in, bias_in});
    const TensorImplPtr& x1 = operands[0];
    const TensorImplPtr& x2 = operands[1];
    const TensorImplPtr& weight = operands[2];
    const TensorImplPtr& bias = operands[3];
    require_bilinear_operands(x1, x2, weight, bias);
    const auto bs = flatten_bilinear(x1->shape(), x2->shape(), weight->shape());
    Shape out_shape = x1->shape();
    out_shape.back() = static_cast<std::int64_t>(bs.Dout);
    OpScopeFull scope{schema_v1.name, x1->device(), x1->dtype(), out_shape};

    const bool has_bias = (bias != nullptr);
    Storage bias_storage = has_bias ? bias->storage() : Storage{CpuStorage{}};

    auto& be = backend::Dispatcher::for_device(x1->device());
    Storage out_storage =
        be.bilinear_layer_forward(x1->storage(), x2->storage(), weight->storage(), bias_storage,
                                  has_bias, x1->shape(), x2->shape(), weight->shape(), x1->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, x1->dtype(),
                                            x1->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        std::vector<TensorImplPtr> ins{x1, x2, weight};
        if (bias)
            ins.push_back(bias);
        trc->on_op_io(ins, out);
    }
    const bool any_grad = x1->requires_grad() || x2->requires_grad() || weight->requires_grad() ||
                          (bias && bias->requires_grad());
    if (!GradMode::is_enabled() || !any_grad)
        return out;

    auto bwd = std::make_shared<BilinearLayerBackward>();
    bwd->orig_x1_shape_ = x1->shape();
    bwd->orig_x2_shape_ = x2->shape();
    // The bias is read by shape alone and is not saved.
    kernel::NaryKernel<BilinearLayerBackward, 4>::wire_autograd(
        std::move(bwd), {x1, x2, weight, bias}, out, {true, true, true, false});
    return out;
}

std::vector<Storage> BilinearLayerBackward::apply(Storage grad_out) {
    const bool has_bias = !input_shapes_[3].empty();

    auto& be = backend::Dispatcher::for_device(device_);
    return be.bilinear_layer_backward(grad_out, saved_inputs_[0], saved_inputs_[1],
                                      saved_inputs_[2], orig_x1_shape_, orig_x2_shape_,
                                      input_shapes_[2], has_bias, dtype_);
}

std::vector<TensorImplPtr> BilinearLayerBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // y[n, o] = x1[n, i] W[o, i, j] x2[n, j] + b[o], with n every leading
    // axis flattened.  Each gradient is the same form with one factor
    // replaced by g: dx1 = (g W) x2, dx2 = (g W)^T x1, dW = g^T (x1 x2^T),
    // db = the sum of g over n.
    const auto& x1 = saved_impl_inputs_[0];
    const auto& x2 = saved_impl_inputs_[1];
    const auto& W = saved_impl_inputs_[2];
    if (!x1 || !x2 || !W)
        ErrorBuilder("bilinear_layer").fail("graph-mode backward is missing a saved input");
    const Shape& ws = input_shapes_[2];
    const std::int64_t o = ws[0], i = ws[1], j = ws[2];
    const std::int64_t n = x1->numel() / i;

    auto g = reshape_op(grad_out, {n, o});
    auto a = reshape_op(x1, {n, i, 1});
    auto b = reshape_op(x2, {n, j, 1});
    auto gw = reshape_op(matmul_op(g, reshape_op(W, {o, i * j})), {n, i, j});
    auto dx1 = reshape_op(matmul_op(gw, b), orig_x1_shape_);
    auto dx2 = reshape_op(matmul_op(permute_op(gw, {0, 2, 1}), a), orig_x2_shape_);
    auto outer = reshape_op(matmul_op(a, permute_op(b, {0, 2, 1})), {n, i * j});
    auto dW = reshape_op(matmul_op(permute_op(g, {1, 0}), outer), {o, i, j});
    auto db = sum_op(g, std::vector<int>{0}, /*keepdims=*/false);
    return {dx1, dx2, dW, db};
}

TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                const TensorImplPtr& x2,
                                const TensorImplPtr& weight,
                                const TensorImplPtr& bias) {
    return BilinearLayerBackward::forward(x1, x2, weight, bias);
}
LUCID_REGISTER_OP(BilinearLayerBackward)

}  // namespace lucid
