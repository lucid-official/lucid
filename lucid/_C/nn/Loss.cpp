// lucid/_C/nn/Loss.cpp
//
// Implementations of MSE, BCE, BCEWithLogits, CrossEntropy, NLL, and Huber
// loss functions with full autograd support.
//
// All forward methods follow the same pattern:
//   1. Validate inputs (shape, dtype, device).
//   2. Delegate the actual computation to IBackend (ForceFP32 schema).
//   3. Construct the output TensorImpl with the reduced shape (scalar for
//      Mean/Sum, per-sample shape for None).
//   4. Wire the backward node via NaryKernel::wire_autograd.
//
// For CrossEntropy and NLL, the backend also returns saved auxiliary tensors
// (softmax probabilities, valid_count) needed by the backward pass; these are
// stored directly in the backward node rather than via saved_inputs_.

#include "Loss.h"

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
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Arith.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/Select.h"

namespace lucid {

namespace {

// Return the output shape after applying the reduction.
// Mean and Sum collapse to a scalar (empty shape); None preserves input shape.
Shape reduced_shape(const Shape& in, Reduction red) {
    if (red == Reduction::None)
        return in;
    return Shape{};
}

// The upstream gradient of a reduced loss, spread back over the elements:
// as is for None, broadcast for Sum, broadcast and divided by the count for
// Mean.  Recorded, for the graph-mode backwards below.
TensorImplPtr
spread_loss_grad(const TensorImplPtr& grad_out, const TensorImplPtr& like, Reduction red) {
    if (red == Reduction::None)
        return grad_out;
    auto spread = broadcast_to_op(grad_out, like->shape());
    if (red == Reduction::Mean) {
        const double count = static_cast<double>(shape_numel(like->shape()));
        spread = mul_op(spread, full_like_op(like, count > 0 ? 1.0 / count : 0.0));
    }
    return spread;
}

}  // namespace

const OpSchema MseLossBackward::schema_v1{"mse_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr MseLossBackward::forward(const TensorImplPtr& input0,
                                       const TensorImplPtr& target0,
                                       Reduction reduction) {
    if (!input0 || !target0)
        ErrorBuilder("mse_loss").fail("null input");
    // Both operands take the schema dtype the kernel templates would
    // have applied — a loss of two integer tensors is not an integer.
    // See promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    const TensorImplPtr target = promote_for_schema(schema_v1, target0);

    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "mse_loss: input/target shape mismatch");
    if (input->dtype() != target->dtype())
        throw DtypeMismatch(std::string(dtype_name(input->dtype())),
                            std::string(dtype_name(target->dtype())), "mse_loss");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));

    Storage out_storage = backend::Dispatcher::for_device(input->device())
                              .mse_loss(input->storage(), target->storage(), input->shape(),
                                        input->dtype(), static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<MseLossBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<MseLossBackward, 2>::wire_autograd(std::move(bwd), {input, target}, out);
    }
    return out;
}

std::vector<TensorImplPtr> MseLossBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // d/dx = 2 (x - t) g and d/dt its negative, recorded, so the residual
    // keeps its dependence on both operands.
    const auto& x = saved_impl_inputs_[0];
    const auto& t = saved_impl_inputs_[1];
    if (!x || !t)
        ErrorBuilder("mse_loss").fail("graph-mode backward is missing its saved inputs");
    auto residual = sub_op(x, t);
    auto dx = mul_op(mul_op(residual, full_like_op(residual, 2.0)),
                     spread_loss_grad(grad_out, x, reduction_));
    return {dx, neg_op(dx)};
}

std::vector<Storage> MseLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_).mse_loss_backward(
        saved_inputs_[0], saved_inputs_[1], grad_out, orig_shape_, dtype_,
        static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr mse_loss_op(const TensorImplPtr& input, const TensorImplPtr& target, int reduction) {
    return MseLossBackward::forward(input, target, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(MseLossBackward)

const OpSchema BCELossBackward::schema_v1{"bce_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCELossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight,
                                       Reduction reduction,
                                       double eps) {
    if (!input || !target || !weight)
        ErrorBuilder("bce_loss")
            .fail("input/target/weight required (pass ones for weight if not used)");
    // The kernels read every operand in the input's dtype; a mismatched one
    // was read as such — a float16 target as float32 bytes, say.
    Validator::pair(input, target, "bce_loss: input/target").same_device().same_dtype();
    Validator::pair(input, weight, "bce_loss: input/weight").same_device().same_dtype();
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_loss: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));
    scope.set_attr("eps", eps);

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_loss(input->storage(), target->storage(), weight->storage(), input->shape(),
                      input->dtype(), eps, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCELossBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCELossBackward, 3>::wire_autograd(std::move(bwd),
                                                              {input, target, weight}, out);
    }
    return out;
}

std::vector<Storage> BCELossBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).bce_loss_backward(
        saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], grad_out, orig_shape_, dtype_, eps_,
        static_cast<int>(reduction_));
}

TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight,
                          int reduction,
                          double eps) {
    return BCELossBackward::forward(input, target, weight, static_cast<Reduction>(reduction), eps);
}
LUCID_REGISTER_OP(BCELossBackward)

const OpSchema BCEWithLogitsBackward::schema_v1{"bce_with_logits", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCEWithLogitsBackward::forward(const TensorImplPtr& input,
                                             const TensorImplPtr& target,
                                             const TensorImplPtr& weight,
                                             const TensorImplPtr& pos_weight,
                                             Reduction reduction) {
    if (!input || !target || !weight || !pos_weight)
        ErrorBuilder("bce_with_logits").fail("input/target/weight/pos_weight required");
    Validator::pair(input, target, "bce_with_logits: input/target").same_device().same_dtype();
    Validator::pair(input, weight, "bce_with_logits: input/weight").same_device().same_dtype();
    Validator::pair(input, pos_weight, "bce_with_logits: input/pos_weight")
        .same_device()
        .same_dtype();
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_with_logits: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_with_logits_loss(input->storage(), target->storage(), weight->storage(),
                                  pos_weight->storage(), input->shape(), weight->shape(),
                                  pos_weight->shape(), input->dtype(), static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCEWithLogitsBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCEWithLogitsBackward, 4>::wire_autograd(
            std::move(bwd), {input, target, weight, pos_weight}, out);
    }
    return out;
}

std::vector<Storage> BCEWithLogitsBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).bce_with_logits_backward(
        saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], saved_inputs_[3], grad_out,
        orig_shape_, dtype_, static_cast<int>(reduction_));
}

TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 int reduction) {
    return BCEWithLogitsBackward::forward(input, target, weight, pos_weight,
                                          static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(BCEWithLogitsBackward)

const OpSchema CrossEntropyBackward::schema_v1{"cross_entropy_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr CrossEntropyBackward::forward(const TensorImplPtr& input,
                                            const TensorImplPtr& target,
                                            const TensorImplPtr& weight_or_null,
                                            Reduction reduction,
                                            double eps,
                                            int ignore_index) {
    if (!input || !target)
        ErrorBuilder("cross_entropy").fail("null input");
    require_min_rank(input->shape(), 2, "cross_entropy: input", "(N, C, *)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())),
                             "cross_entropy: input/target");
    if (weight_or_null)
        Validator::pair(input, weight_or_null, "cross_entropy: input/weight")
            .same_device()
            .same_dtype();

    const int N = static_cast<int>(input->shape()[0]);

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));
    scope.set_attr("ignore_index", static_cast<std::int64_t>(ignore_index));

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    const Storage* weight_storage = weight_or_null ? &weight_or_null->storage() : nullptr;
    auto result = backend::Dispatcher::for_device(input->device())
                      .cross_entropy_loss(input->storage(), target->storage(), weight_storage,
                                          input->shape(), target->shape(), input->dtype(), eps,
                                          ignore_index, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(result.output), out_shape, input->dtype(),
                                            input->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        if (weight_or_null)
            trc->on_op_io({input, target, weight_or_null}, out);
        else
            trc->on_op_io({input, target}, out);
    }

    {
        auto bwd = std::make_shared<CrossEntropyBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_softmax_ = std::move(result.saved_aux);
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ = std::move(result.valid_count);
        kernel::NaryKernel<CrossEntropyBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                                   false);
    }
    return out;
}

std::vector<Storage> CrossEntropyBackward::apply(Storage grad_out) {
    const Storage* weight_storage = has_weight_ ? &saved_weight_ : nullptr;
    auto dx = backend::Dispatcher::for_device(device_).cross_entropy_backward(
        saved_softmax_, saved_target_, weight_storage, saved_valid_count_, grad_out,
        orig_input_shape_, dtype_, ignore_index_, static_cast<int>(reduction_));
    return {std::move(dx)};
}

TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                               const TensorImplPtr& target,
                               const TensorImplPtr& weight_or_null,
                               int reduction,
                               double eps,
                               int ignore_index) {
    return CrossEntropyBackward::forward(input, target, weight_or_null,
                                         static_cast<Reduction>(reduction), eps, ignore_index);
}
LUCID_REGISTER_OP(CrossEntropyBackward)

const OpSchema NLLLossBackward::schema_v1{"nll_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr NLLLossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight_or_null,
                                       Reduction reduction,
                                       int ignore_index) {
    if (!input || !target)
        ErrorBuilder("nll_loss").fail("null input");
    require_min_rank(input->shape(), 2, "nll_loss: input", "(N, C, *)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())), "nll_loss: input/target");
    if (weight_or_null)
        Validator::pair(input, weight_or_null, "nll_loss: input/weight").same_device().same_dtype();

    const int N = static_cast<int>(input->shape()[0]);

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));
    scope.set_attr("ignore_index", static_cast<std::int64_t>(ignore_index));

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    const Storage* weight_storage = weight_or_null ? &weight_or_null->storage() : nullptr;
    auto result =
        backend::Dispatcher::for_device(input->device())
            .nll_loss(input->storage(), target->storage(), weight_storage, input->shape(),
                      target->shape(), input->dtype(), ignore_index, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(result.output), out_shape, input->dtype(),
                                            input->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        if (weight_or_null)
            trc->on_op_io({input, target, weight_or_null}, out);
        else
            trc->on_op_io({input, target}, out);
    }

    {
        auto bwd = std::make_shared<NLLLossBackward>();
        bwd->reduction_ = reduction;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ = std::move(result.valid_count);
        kernel::NaryKernel<NLLLossBackward, 1>::wire_autograd(std::move(bwd), {input}, out, false);
    }
    return out;
}

std::vector<Storage> NLLLossBackward::apply(Storage grad_out) {
    const Storage* weight_storage = has_weight_ ? &saved_weight_ : nullptr;
    auto dx = backend::Dispatcher::for_device(device_).nll_loss_backward(
        saved_target_, weight_storage, saved_valid_count_, grad_out, orig_input_shape_, dtype_,
        ignore_index_, static_cast<int>(reduction_));
    return {std::move(dx)};
}

TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight_or_null,
                          int reduction,
                          int ignore_index) {
    return NLLLossBackward::forward(input, target, weight_or_null,
                                    static_cast<Reduction>(reduction), ignore_index);
}
LUCID_REGISTER_OP(NLLLossBackward)

const OpSchema HuberLossBackward::schema_v1{"huber_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr HuberLossBackward::forward(const TensorImplPtr& input0,
                                         const TensorImplPtr& target0,
                                         double delta,
                                         Reduction reduction) {
    if (!input0 || !target0)
        ErrorBuilder("huber_loss").fail("null input");
    // Both operands take the schema dtype the kernel templates would
    // have applied — a loss of two integer tensors is not an integer.
    // See promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    const TensorImplPtr target = promote_for_schema(schema_v1, target0);

    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "huber_loss: input/target shape mismatch");
    Validator::pair(input, target, "huber_loss").same_device().same_dtype();
    if (delta <= 0.0)
        ErrorBuilder("huber_loss").invalid_argument("delta must be positive");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};
    scope.set_attr("reduction", static_cast<std::int64_t>(reduction));
    scope.set_attr("delta", delta);

    Storage out_storage = backend::Dispatcher::for_device(input->device())
                              .huber_loss(input->storage(), target->storage(), input->shape(),
                                          input->dtype(), delta, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<HuberLossBackward>();
        bwd->reduction_ = reduction;
        bwd->delta_ = delta;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<HuberLossBackward, 2>::wire_autograd(std::move(bwd), {input, target},
                                                                out);
    }
    return out;
}

std::vector<TensorImplPtr> HuberLossBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // d/dx = r inside the band |r| < delta and delta * sign(r) outside it,
    // r = x - t; d/dt its negative.  Recorded, so the quadratic part keeps
    // its unit curvature and the linear part none.
    const auto& x = saved_impl_inputs_[0];
    const auto& t = saved_impl_inputs_[1];
    if (!x || !t)
        ErrorBuilder("huber_loss").fail("graph-mode backward is missing its saved inputs");
    auto residual = sub_op(x, t);
    auto delta = full_like_op(residual, delta_);
    auto inside = less_op(abs_op(residual), delta);
    auto slope = where_op(inside, residual, mul_op(sign_op(residual), delta));
    auto dx = mul_op(slope, spread_loss_grad(grad_out, x, reduction_));
    return {dx, neg_op(dx)};
}

std::vector<Storage> HuberLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_).huber_loss_backward(
        saved_inputs_[0], saved_inputs_[1], grad_out, orig_shape_, dtype_, delta_,
        static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                            const TensorImplPtr& target,
                            double delta,
                            int reduction) {
    return HuberLossBackward::forward(input, target, delta, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(HuberLossBackward)

// ── CTC Loss ──────────────────────────────────────────────────────────────────

TensorImplPtr ctc_loss_op(const TensorImplPtr& log_probs_in,
                          const TensorImplPtr& targets,
                          const TensorImplPtr& input_lengths,
                          const TensorImplPtr& target_lengths,
                          int blank,
                          bool zero_infinity) {
    Validator::input(log_probs_in, "ctc_loss.log_probs").non_null();
    // The autocast row (float32 on both devices) casts a half input first.
    // Without it a bfloat16 ``log_softmax`` output under a CPU autocast
    // reached the kernel, which has no half path.  An integer input is not
    // promoted, so the check below still refuses it.
    const TensorImplPtr log_probs =
        is_floating_point(log_probs_in->dtype())
            ? promote_for_schema(CtcLossBackward::schema_v1, log_probs_in)
            : log_probs_in;
    Validator::input(log_probs, "ctc_loss.log_probs").float_only();
    Validator::input(targets, "ctc_loss.targets").non_null();
    Validator::input(input_lengths, "ctc_loss.input_lengths").non_null();
    Validator::input(target_lengths, "ctc_loss.target_lengths").non_null();

    const auto& lp_shape = log_probs->shape();
    if (lp_shape.size() != 3)
        ErrorBuilder("ctc_loss").fail("log_probs must be 3-D (T, N, C)");
    // Rank was checked; extents were not.  The forward-backward recursion
    // indexes the first time step before it loops, so a zero-length T read
    // a lattice that was never allocated — SIGSEGV on an empty (0, N, C)
    // input.  A zero alphabet has no blank to emit, and a zero batch has
    // nothing to reduce over.
    if (lp_shape[0] <= 0)
        ErrorBuilder("ctc_loss").fail("log_probs has no time steps (T = 0)");
    if (lp_shape[1] <= 0)
        ErrorBuilder("ctc_loss").fail("log_probs has an empty batch (N = 0)");
    if (lp_shape[2] <= 0)
        ErrorBuilder("ctc_loss").fail("log_probs has an empty alphabet (C = 0)");

    const int N = static_cast<int>(lp_shape[1]);
    Shape out_shape{static_cast<std::int64_t>(N)};

    auto& be = backend::Dispatcher::for_device(log_probs->device());
    Storage out = be.ctc_loss_forward(log_probs->storage(), targets->storage(),
                                      input_lengths->storage(), target_lengths->storage(), lp_shape,
                                      blank, zero_infinity, log_probs->dtype());

    auto result = std::make_shared<TensorImpl>(std::move(out), out_shape, log_probs->dtype(),
                                               log_probs->device(), false);
    // The forward recorded no graph, so every loss built on it trained
    // nothing: the per-sample losses came back detached from log_probs.
    auto bwd = std::make_shared<CtcLossBackward>();
    bwd->lp_shape_ = lp_shape;
    bwd->blank_ = blank;
    bwd->zero_infinity_ = zero_infinity;
    bwd->saved_targets_ = targets->storage();
    bwd->saved_input_lengths_ = input_lengths->storage();
    bwd->saved_target_lengths_ = target_lengths->storage();
    kernel::NaryKernel<CtcLossBackward, 1>::wire_autograd(std::move(bwd), {log_probs}, result);
    return result;
}

const OpSchema CtcLossBackward::schema_v1{"ctc_loss", 1, AmpPolicy::ForceFP32, true};

std::vector<Storage> CtcLossBackward::apply(Storage grad_out) {
    return {backend::Dispatcher::for_device(device_).ctc_loss_backward(
        saved_inputs_[0], saved_targets_, saved_input_lengths_, saved_target_lengths_, grad_out,
        lp_shape_, blank_, zero_infinity_, dtype_)};
}
LUCID_REGISTER_OP(CtcLossBackward)

}  // namespace lucid
