// lucid/_C/ops/utils/Select.cpp
//
// Implements where, masked_fill, roll, gather, diagonal, flip, and
// masked_select with autograd where applicable.
//
// Backward nodes:
//   WhereBackward      — mask grad_out with `cond` to obtain dx (x-branch)
//                        and with ~cond to obtain dy (y-branch); cond itself
//                        is non-differentiable.
//   MaskedFillBackward — pass grad_out only through positions where mask is
//                        false (positions that kept the original value).
//   RollBackward       — apply an inverse circular roll by negating all shifts
//                        along the same axes; this exactly inverts a circular
//                        shift of +k with one of -k.
//   GatherBackward     — scatter-add grad_out back to source positions in the
//                        original tensor using the saved index storage.
//   DiagonalBackward   — scatter the length-L diagonal gradient back into a
//                        zero-initialised buffer at the correct (row, col)
//                        positions determined by offset, axis1, and axis2.
//
// where, masked_fill and gather read a selector — the condition, the mask, the
// index — that is not one of their differentiable inputs, so they guard its
// version themselves; the inputs they differentiate are read by shape alone
// and are not refused.  The remaining ops use the single-input FuncOp<1>
// template for simplicity.

#include "Select.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AutogradNode.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // detail::broadcast_shapes
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Mul.h"
#include "../bfunc/_BinaryOp.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Reductions.h"
#include "../ufunc/Transpose.h"
#include "Layout.h"  // broadcast_to_op
#include "Pad.h"
#include "ScalarAxis.h"
#include "View.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;
using utils_detail::wrap_axis;

// Compute the gradient contribution for one branch of where_op.
//
// When `true_branch` is true, this returns grad * cond (the gradient flowing
// to `x`, the tensor chosen where cond is true).  When false, it returns
// grad * ~cond (the gradient flowing to `y`, chosen where cond is false).
// The dispatcher handles both CPU (same-shape elementwise) and GPU (broadcast).
Storage where_branch_storage(const Storage& grad,
                             const Storage& cond,
                             const Shape& shape,
                             Dtype dt,
                             Device device,
                             bool true_branch) {
    return backend::Dispatcher::for_device(device).where_branch(grad, cond, shape, dt, true_branch);
}

// Scatter-add the output gradient back to input positions using the same
// index tensor that was passed to gather_op.  Each element of grad at position
// p is accumulated (scatter-add) into the input gradient at the position
// indices[p] along `axis`.  This is the adjoint of the gather operation.
Storage gather_backward_storage(const Storage& grad,
                                const Storage& indices,
                                const Shape& input_shape,
                                const Shape& output_shape,
                                int axis,
                                Dtype index_dtype,
                                Dtype dt,
                                Device device) {
    return backend::Dispatcher::for_device(device).gather_backward(
        grad, indices, input_shape, output_shape, axis, index_dtype, dt);
}

// Scatter the length-L diagonal gradient back into a zero-initialised buffer
// of `input_shape`.  The offset, axis1, and axis2 parameters are used to
// identify which (row, col) pairs along the two chosen axes correspond to
// the diagonal elements so that each gradient value lands at the right position.
Storage diagonal_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int offset,
                                  int axis1,
                                  int axis2,
                                  Dtype dt,
                                  Device device) {
    return backend::Dispatcher::for_device(device).diagonal_backward(
        grad, input_shape, output_shape, offset, axis1, axis2, dt);
}

// Backward node for where_op.
//
// Invariants:
//   cond_       — a copy of the condition storage, used to route gradients.
//   shape_      — the output (and condition) shape.
//   cond_tensor_ — weak ref to the condition, for version checking.
//
// Backward: grad_out * cond → dx,  grad_out * ~cond → dy.
// Sum a gradient over the axes broadcasting expanded, back to ``shape`` —
// the graph-mode counterpart of ``reduce_grad_to_shape``, as
// ``BinaryKernel::reduce_impl_to_shape`` is for the binary ops.
TensorImplPtr sum_to_shape(const TensorImplPtr& grad, const Shape& shape) {
    if (grad->shape() == shape)
        return grad;
    const auto& gs = grad->shape();
    const int ng = static_cast<int>(gs.size());
    const int nt = static_cast<int>(shape.size());
    std::vector<int> axes;
    for (int i = 0; i < ng - nt; ++i)
        axes.push_back(i);
    for (int i = 0; i < nt; ++i)
        if (shape[static_cast<std::size_t>(i)] == 1 &&
            gs[static_cast<std::size_t>(i + ng - nt)] != 1)
            axes.push_back(i + ng - nt);
    auto reduced = axes.empty() ? grad : sum_op(grad, axes, false);
    return reduced->shape() == shape ? reduced : reshape_op(reduced, shape);
}

class WhereBackward : public AutogradNode<WhereBackward, 2> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Storage cond_;
    Shape shape_;
    // Each operand's own shape.  On the GPU MLX broadcasts them natively,
    // so a branch's gradient has the output's shape and must be summed back
    // to the operand's — a ``(1, 3)`` operand of a ``(2, 3)`` result used to
    // receive the first row, not the column sums.  The CPU expands them
    // first, and there the shapes already agree.
    Shape cond_shape_;
    Shape x_shape_;
    Shape y_shape_;
    std::weak_ptr<TensorImpl> cond_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage dx;
        if (needs_input_grad(0)) {
            dx = where_branch_storage(grad_out, cond_, shape_, dtype_, device_, true);
            if (x_shape_ != shape_)
                dx = reduce_grad_to_shape(dx, shape_, x_shape_, dtype_, device_);
        }
        Storage dy;
        if (needs_input_grad(1)) {
            dy = where_branch_storage(grad_out, cond_, shape_, dtype_, device_, false);
            if (y_shape_ != shape_)
                dy = reduce_grad_to_shape(dy, shape_, y_shape_, dtype_, device_);
        }
        return {std::move(dx), std::move(dy)};
    }

    // Graph mode: the same routing, built from ops, so the gradient is
    // differentiable in turn.
    //
    // It was written once before and reverted: ``cdist``'s second derivative
    // came back wrong, and ``where`` took the blame.  The fault was upstream.
    // ``cdist`` is ``where(sq == 0, 0, sqrt(sq))``, this node pins neither
    // branch, so ``sqrt``'s output was dropped — and its graph-mode backward
    // rebuilt it as a leaf, turning ``g / 2y`` into a formula with a
    // constant ``y``.  The rebuild now keeps the output's grad_fn
    // (``UnaryKernel::apply_for_graph``), which is what made this safe.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        auto cond = std::make_shared<TensorImpl>(cond_, cond_shape_, Dtype::Bool, device_, false);
        auto zero = zeros_like_op(grad_out);
        return {sum_to_shape(where_op(cond, grad_out, zero), x_shape_),
                sum_to_shape(where_op(cond, zero, grad_out), y_shape_)};
    }

    // The condition is the one tensor read by value; ``x`` and ``y`` only
    // route the gradient, so a write into either is not refused.
    void validate_versions() override { check_saved_version(cond_tensor_, 0, schema_v1.name, 0); }
    void release_own_saved() { clear_saved(cond_); }
};

const OpSchema WhereBackward::schema_v1{"where", 1, AmpPolicy::KeepInput, true};

// Backward node for masked_fill_op.
//
// Invariants:
//   mask_  — a copy of the boolean mask storage.
//   shape_ — tensor shape (same for input and output).
//
// Backward: pass gradient only through positions where mask was false (the
// positions whose values were kept from the original input).  Positions where
// mask is true received the fill constant, so their gradient is zero.
class MaskedFillBackward : public AutogradNode<MaskedFillBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Storage mask_;
    Dtype mask_dtype_ = Dtype::Bool;
    Shape shape_;
    std::weak_ptr<TensorImpl> mask_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        // false_branch passes gradient through where mask is false.
        return {where_branch_storage(grad_out, mask_, shape_, dtype_, device_, false)};
    }

    // where(mask, 0, g), recorded: the filled positions took a constant and
    // pass nothing back, at any order.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        TensorImplPtr mask =
            std::make_shared<TensorImpl>(mask_, shape_, mask_dtype_, device_, false);
        if (mask_dtype_ != Dtype::Bool)
            mask = astype_op(mask, Dtype::Bool);
        return {where_op(mask, zeros_like_op(grad_out), grad_out)};
    }

    // Only the mask is read by value.
    void validate_versions() override { check_saved_version(mask_tensor_, 0, schema_v1.name, 1); }
    void release_own_saved() { clear_saved(mask_); }
};

const OpSchema MaskedFillBackward::schema_v1{"masked_fill", 1, AmpPolicy::KeepInput, true};

// Backward node for roll_op.
//
// Backward formula: roll the gradient by the negated shifts along the same
// axes.  A circular shift of +k is exactly inverted by a shift of -k.
class RollBackward : public FuncOp<RollBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::int64_t> shifts_;
    std::vector<int> axes_;

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<std::int64_t> inv_shifts;
        inv_shifts.reserve(shifts_.size());
        for (auto s : shifts_)
            inv_shifts.push_back(-s);
        return {backend::Dispatcher::for_device(device_).roll(grad_out, out_shape_, dtype_,
                                                              inv_shifts, axes_)};
    }

    // Graph-mode: rolling back by the same shifts.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        std::vector<std::int64_t> inv_shifts;
        inv_shifts.reserve(shifts_.size());
        for (auto shift : shifts_)
            inv_shifts.push_back(-shift);
        return {roll_op(grad_out, inv_shifts, axes_)};
    }
};

const OpSchema RollBackward::schema_v1{"roll", 1, AmpPolicy::KeepInput, true};

// Backward node for gather_op.
//
// Invariants:
//   indices_      — saved index storage from the forward call.
//   input_shape_  — shape of the source tensor `a`.
//   output_shape_ — shape of the gathered output.
//   axis_         — axis along which gathering was performed.
//   index_dtype_  — dtype of the index tensor (I32 or I64).
//
// Backward formula: scatter-add grad_out back to the input positions indicated
// by indices_.  This is the adjoint of gather.
class GatherBackward : public AutogradNode<GatherBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Storage indices_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;
    Dtype index_dtype_ = Dtype::I64;
    std::weak_ptr<TensorImpl> indices_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {gather_backward_storage(grad_out, indices_, input_shape_, output_shape_, axis_,
                                        index_dtype_, dtype_, device_)};
    }

    // Graph-mode: the adjoint of a gather is a scatter-add.
    //
    // Each output element came from one input position, so the gradient
    // goes back to that position; duplicates in the index accumulate,
    // which is what makes it an *add* rather than a write.
    //
    // ``scatter_add_op`` already supports create_graph, so composing
    // through it keeps the result differentiable in turn — the whole
    // point of a graph-mode formula.  This unblocks ``cross_entropy``,
    // ``nll_loss`` and ``index_select`` as well as ``gather`` itself:
    // sixteen public symbols behind one missing override.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        auto indices =
            std::make_shared<TensorImpl>(indices_, output_shape_, index_dtype_, device_, false);
        auto zero = zeros_op(input_shape_, dtype_, device_);
        return {scatter_add_op(zero, indices, grad_out, axis_)};
    }

    // Only the index is read by value.
    void validate_versions() override {
        check_saved_version(indices_tensor_, 0, schema_v1.name, 1);
    }
    void release_own_saved() { clear_saved(indices_); }
};

const OpSchema GatherBackward::schema_v1{"gather", 1, AmpPolicy::KeepInput, true};

// Backward node for diagonal_op.
//
// Backward formula: scatter the length-L diagonal gradient back into a zero
// tensor of the original input shape at the positions determined by offset,
// axis1, and axis2.
class DiagonalBackward : public FuncOp<DiagonalBackward, 1> {
public:
    static const OpSchema schema_v1;

    int offset_ = 0;
    int axis1_ = 0;
    int axis2_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        return {diagonal_backward_storage(grad_out, input_shapes_[0], out_shape_, offset_, axis1_,
                                          axis2_, dtype_, device_)};
    }

    // The same placement, recorded.  The diagonal's k-th element sits in
    // column k + max(offset, 0) of the (axis1, axis2) plane, so the gradient
    // is padded out to a full row indexed by column, broadcast down the
    // rows, and kept only where the offset identity has its ones; then the
    // two axes go back where the input had them.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const Shape& in = input_shapes_[0];
        const int nd = static_cast<int>(in.size());
        const std::int64_t rows = in[static_cast<std::size_t>(axis1_)];
        const std::int64_t cols = in[static_cast<std::size_t>(axis2_)];
        const std::int64_t length = grad_out->shape().back();
        const std::int64_t lead = offset_ > 0 ? offset_ : 0;

        std::vector<std::pair<std::int64_t, std::int64_t>> widths(grad_out->shape().size(), {0, 0});
        widths.back() = {lead, cols - lead - length};
        auto by_column = pad_op(grad_out, widths, 0.0);  // (..., cols)
        Shape plane(grad_out->shape().begin(), grad_out->shape().end() - 1);
        plane.push_back(rows);
        plane.push_back(cols);
        auto spread = broadcast_to_op(unsqueeze_op(by_column, nd - 2), plane);
        auto placed = mul_op(spread, eye_op(rows, cols, offset_, dtype_, device_));

        // ``placed`` holds the input's other axes in order, then axis1, axis2.
        std::vector<int> order;
        for (int d = 0; d < nd; ++d)
            if (d != axis1_ && d != axis2_)
                order.push_back(d);
        order.push_back(axis1_);
        order.push_back(axis2_);
        std::vector<int> back(static_cast<std::size_t>(nd));
        for (int i = 0; i < nd; ++i)
            back[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = i;
        return {permute_op(placed, back)};
    }
};

const OpSchema DiagonalBackward::schema_v1{"diagonal", 1, AmpPolicy::KeepInput, true};

// Build and attach WhereBackward to `out`.  Gradient edges connect only to
// `x` and `y` — the condition tensor `cond` is treated as a non-differentiable
// selector.  The WhereBackward node retains a copy of the condition storage
// to perform the masking during the backward pass.
TensorImplPtr attach_where_grad(const TensorImplPtr& cond,
                                const TensorImplPtr& x,
                                const TensorImplPtr& y,
                                TensorImplPtr out) {
    const bool needs_grad = GradMode::is_enabled() && (x->requires_grad() || y->requires_grad());
    if (!needs_grad)
        return out;

    auto bwd = std::make_shared<WhereBackward>();
    bwd->cond_ = cond->storage();
    bwd->shape_ = out->shape();
    bwd->cond_shape_ = cond->shape();
    bwd->x_shape_ = x->shape();
    bwd->y_shape_ = y->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->cond_tensor_ = cond;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(x), x->grad_output_nr()),
                                          Edge(detail::ensure_grad_fn(y), y->grad_output_nr())});
    bwd->set_saved_versions({cond->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

// Build and attach MaskedFillBackward to `out`.  Only the data input `a`
// carries a gradient edge; the boolean mask is a non-differentiable selector.
// The mask storage is retained so that the backward can zero gradient entries
// at positions where the fill constant was written.
TensorImplPtr
attach_masked_fill_grad(const TensorImplPtr& a, const TensorImplPtr& mask, TensorImplPtr out) {
    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<MaskedFillBackward>();
    bwd->mask_ = mask->storage();
    bwd->mask_dtype_ = mask->dtype();
    bwd->shape_ = out->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->mask_tensor_ = mask;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), a->grad_output_nr())});
    bwd->set_saved_versions({mask->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

// Generic helper to wire a pre-built, fully initialised single-input backward
// node onto `out`.  Used by roll_op and diagonal_op, which construct their
// backward nodes inline before calling this function.
template <class Derived>
TensorImplPtr
attach_unary_grad(const TensorImplPtr& a, TensorImplPtr out, std::shared_ptr<Derived> bwd) {
    kernel::NaryKernel<Derived, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(WhereBackward)
LUCID_REGISTER_OP(MaskedFillBackward)
LUCID_REGISTER_OP(RollBackward)
LUCID_REGISTER_OP(GatherBackward)
LUCID_REGISTER_OP(DiagonalBackward)
// FlipBackward register lives after the class definition near flip_op below.

std::tuple<TensorImplPtr, TensorImplPtr, TensorImplPtr>
align_where_devices(const TensorImplPtr& cond, const TensorImplPtr& x, const TensorImplPtr& y) {
    if (cond->device() == x->device() && x->device() == y->device())
        return {cond, x, y};
    auto lift = [](const TensorImplPtr& t) {
        return (t->device() == Device::CPU && t->shape().empty()) ? to_device_op(t, Device::GPU)
                                                                  : t;
    };
    auto c_out = lift(cond);
    auto x_out = lift(x);
    auto y_out = lift(y);
    Validator::pair(x_out, y_out, "where: x/y").same_device();
    Validator::pair(x_out, c_out, "where: x/condition").same_device();
    return {c_out, x_out, y_out};
}

}  // namespace

// Select elements from `x` (where cond is true) or `y` (where cond is false).
// Validates that x and y share dtype and device, and that cond is on the same
// device.  On the CPU backend, all three tensors must have identical shapes;
// on the GPU backend the MLX backend handles implicit broadcasting, and the
// actual output shape is read back from the MLX array after execution.
// Attaches WhereBackward only when x or y requires a gradient.
TensorImplPtr
where_op(const TensorImplPtr& cond_in, const TensorImplPtr& x_in, const TensorImplPtr& y_in) {
    if (!cond_in || !x_in || !y_in)
        ErrorBuilder("where").fail("null input");
    Validator::pair(x_in, y_in, "where: x/y").same_dtype();
    const auto [cond, x, y] = align_where_devices(cond_in, x_in, y_in);
    const Dtype dt = x->dtype();
    const Device device = x->device();

    // All three operands broadcast against each other, as they do for every
    // other elementwise op.  The CPU kernels index their operands linearly
    // and cannot broadcast, so the expansion is materialised there; MLX does
    // it natively, so the GPU operands pass through untouched and no node is
    // added to a trace.  Routing the backward through the broadcast tensors
    // is what makes the gradient correct — broadcast_to's own backward sums
    // over the axes it expanded.
    //
    // The guard this replaces compared cond against x but never x against y,
    // so a mismatched y was read past its end by the CPU kernel instead of
    // being rejected.
    Shape out_shape = x->shape();
    if (!(cond->shape() == x->shape() && x->shape() == y->shape())) {
        out_shape = detail::broadcast_shapes(detail::broadcast_shapes(cond->shape(), x->shape()),
                                             y->shape());
    }
    TensorImplPtr cond_b = cond;
    TensorImplPtr x_b = x;
    TensorImplPtr y_b = y;
    if (device == Device::CPU) {
        if (cond->shape() != out_shape)
            cond_b = broadcast_to_op(cond, out_shape);
        if (x->shape() != out_shape)
            x_b = broadcast_to_op(x, out_shape);
        if (y->shape() != out_shape)
            y_b = broadcast_to_op(y, out_shape);
    }

    OpScopeFull scope{"where", device, dt, out_shape};
    auto out_storage = backend::Dispatcher::for_device(device).where_op(
        cond_b->storage(), x_b->storage(), y_b->storage(), x_b->shape(), dt);
    Shape result_shape;
    if (device == Device::GPU) {
        // Read the actual output shape from the MLX array (may have been broadcast).
        const auto& gs = storage_gpu(out_storage);
        result_shape = mlx_shape_to_lucid(gs.arr->shape());
    } else {
        result_shape = out_shape;
    }
    auto result = fresh(std::move(out_storage), std::move(result_shape), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({cond_b, x_b, y_b}, result);
    }
    return attach_where_grad(cond_b, x_b, y_b, std::move(result));
}

// Replace positions of `a` where `mask` is true with scalar `value`.  Both
// tensors must have identical shapes.  Delegates to the backend dispatcher,
// then attaches MaskedFillBackward so that gradients flow back only through
// the positions that kept the original `a` value (mask == false).
TensorImplPtr masked_fill_op(const TensorImplPtr& a, const TensorImplPtr& mask, double value) {
    if (!a || !mask)
        ErrorBuilder("masked_fill").fail("null input");
    if (a->shape() != mask->shape())
        throw ShapeMismatch(a->shape(), mask->shape(), "masked_fill");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"masked_fill", device, dt, a->shape()};
    scope.set_attr("fill_value", value);
    auto out_storage = backend::Dispatcher::for_device(device).masked_fill(
        a->storage(), mask->storage(), a->shape(), dt, value);
    auto result = fresh(std::move(out_storage), a->shape(), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, mask}, result);
    }
    return attach_masked_fill_grad(a, mask, std::move(result));
}

// Circularly shift `a` by shifts[i] positions along axes[i].  Validates that
// shifts and axes have equal length, normalises each axis value (negative
// wrapping), then dispatches to the backend.  RollBackward is wired with
// the normalised axes and original shifts so the backward can negate each
// shift to produce the exact inverse rotation.
TensorImplPtr
roll_op(const TensorImplPtr& a, std::vector<std::int64_t> shifts, std::vector<int> axes) {
    Validator::input(a, "roll.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"roll", device, dt, a->shape()};
    if (shifts.size() != axes.size())
        ErrorBuilder("roll").fail("shifts and axes must have equal length");
    const std::size_t ndim = a->shape().size();
    for (std::size_t i = 0; i < axes.size(); ++i)
        axes[i] = wrap_axis(axes[i], static_cast<int>(ndim), "roll");
    {
        std::vector<std::int64_t> axes_attr(axes.begin(), axes.end());
        scope.set_attr("shifts", shifts);
        scope.set_attr("axes", axes_attr);
    }
    auto out_storage =
        backend::Dispatcher::for_device(device).roll(a->storage(), a->shape(), dt, shifts, axes);
    auto result = fresh(std::move(out_storage), a->shape(), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    auto bwd = std::make_shared<RollBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->shifts_ = std::move(shifts);
    bwd->axes_ = std::move(axes);
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

namespace {

// A 0-d source has one notional axis.  The result takes the index's shape,
// as on any rank: 0-d for a 0-d index, (n,) for a 1-d one.  Null when ``a``
// is not 0-d or the index is of rank >= 2, which gather_op's rank check
// refuses.
TensorImplPtr gather_from_scalar(const TensorImplPtr& a, const TensorImplPtr& indices, int axis) {
    if (!a->shape().empty() || indices->shape().size() > 1)
        return nullptr;
    wrap_dim(axis, 0, "gather");
    auto out = gather_op(reshape_op(a, {1}), scalar_as_1d(indices), 0);
    return indices->shape().empty() ? reshape_op(out, {}) : out;
}

}  // namespace

// Gather values from `a` at positions given by `indices` along `axis`.  Both
// tensors must have the same rank.  The output shape equals indices.shape.
// Attaches GatherBackward (a scatter-add over indices) only when `a` requires
// a gradient; `indices` itself is always treated as non-differentiable.
TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis) {
    if (!a || !indices)
        ErrorBuilder("gather").fail("null input");
    // Checked, not assumed: a CPU index beside a metal source reached a
    // backend that read the wrong storage kind and threw bad_variant_access.
    Validator::pair(a, indices, "gather").same_device();
    if (auto out = gather_from_scalar(a, indices, axis))
        return out;
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"gather", device, dt, indices->shape()};
    if (a->shape().size() != indices->shape().size())
        throw ShapeMismatch(a->shape(), indices->shape(),
                            "gather: a and indices must have same rank");
    // Index dtype has to be validated here, not in the backends: the CPU
    // kernel happened to reject a float index with NotImplementedError while
    // the GPU one silently reinterpreted it and returned plausible-looking
    // garbage.  One check at the op layer keeps the two honest.
    switch (indices->dtype()) {
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
    case Dtype::I64:
        break;
    default:
        ErrorBuilder("gather").fail("indices must be an integer tensor, got " +
                                    std::string(dtype_name(indices->dtype())));
    }
    const std::size_t ndim = a->shape().size();
    int ax = wrap_axis(axis, static_cast<int>(ndim), "gather");
    // Any index into an empty axis is out of range, and saying so needs only
    // the shapes — no read of the indices, so no host sync on Metal, which
    // otherwise answered with whatever MLX gathers from nothing.  The CPU
    // bounds-checks every index and already refused.
    if (a->shape()[static_cast<std::size_t>(ax)] == 0 && shape_numel(indices->shape()) > 0)
        ErrorBuilder("gather").index_error("index out of range: dimension " + std::to_string(ax) +
                                           " has size 0");
    Shape out_shape = indices->shape();
    auto out_storage = backend::Dispatcher::for_device(device).gather(
        a->storage(), indices->storage(), a->shape(), out_shape, ax, indices->dtype(), dt);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    // 3.5 Phase 1.3: explicit trace I/O — autograd wiring is gated by
    // ``a->requires_grad()`` (and skipped when the index tensor is the
    // grad-bearing one anyway), so the tracer would otherwise see the
    // op header with empty inputs.  We also push the attribute ``axis``
    // so the MPSGraph emitter can wire ``gatherWithUpdatesTensor:`` to
    // the right axis.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, indices}, result);
        trc->on_op_attr("axis", static_cast<std::int64_t>(ax));
    }
    if (GradMode::is_enabled() && a->requires_grad()) {
        auto bwd = std::make_shared<GatherBackward>();
        bwd->indices_ = indices->storage();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = result->shape();
        bwd->axis_ = ax;
        bwd->dtype_ = dt;
        bwd->index_dtype_ = indices->dtype();
        bwd->device_ = device;
        bwd->indices_tensor_ = indices;
        bwd->set_next_edges(
            std::vector<Edge>{Edge(detail::ensure_grad_fn(a), a->grad_output_nr())});
        bwd->set_saved_versions({indices->version()});
        result->set_grad_fn(std::move(bwd));
        result->set_leaf(false);
        result->set_requires_grad(true);
    }
    return result;
}

// Extract the k-th diagonal of `a` along (axis1, axis2).  Validates that the
// input is at least 2-D, normalises both axes, and enforces axis1 < axis2 via
// a swap.  Computes the diagonal length L = max(0, min(M - r0, N - c0)) where
// M=shape[a1], N=shape[a2] and (r0, c0) are the starting row/col for offset.
// The output shape consists of all non-axis dimensions followed by L.
// Attaches DiagonalBackward to scatter the gradient back to the correct
// positions in a zero tensor of the input shape.
TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2) {
    Validator::input(a, "diagonal.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"diagonal", device, dt, a->shape()};
    const std::size_t ndim = a->shape().size();
    if (ndim < 2)
        ErrorBuilder("diagonal").fail("input must be ≥2-D");
    int a1 = wrap_axis(axis1, static_cast<int>(ndim), "diagonal");
    int a2 = wrap_axis(axis2, static_cast<int>(ndim), "diagonal");
    if (a1 == a2)
        ErrorBuilder("diagonal").fail("axis1 and axis2 must differ");
    // Kept in ascending order.  Swapping the axes mirrors the matrix, so the
    // same diagonal sits at the negated offset: (i, i + k) seen from the
    // other side is (i + k, i).
    if (a1 > a2) {
        std::swap(a1, a2);
        offset = -offset;
    }
    scope.set_attr("offset", static_cast<std::int64_t>(offset));
    scope.set_attr("axis1", static_cast<std::int64_t>(a1));
    scope.set_attr("axis2", static_cast<std::int64_t>(a2));

    const std::int64_t M = a->shape()[a1];
    const std::int64_t N = a->shape()[a2];
    // Compute the starting (row, col) corner for the requested diagonal.
    // A positive offset starts at column `offset`, row 0.
    // A negative offset starts at row `-offset`, column 0.
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    // L is the number of elements on the diagonal; 0 if it falls outside the matrix.
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    // Output shape: all dimensions except a1 and a2, then L.
    Shape out_shape;
    for (std::size_t d = 0; d < ndim; ++d) {
        if ((int)d == a1 || (int)d == a2)
            continue;
        out_shape.push_back(a->shape()[d]);
    }
    out_shape.push_back(L);
    TensorImplPtr result;
    if (a->views_by_stride()) {
        // On the CPU the diagonal is a view: one step along it moves one
        // element down both axes, starting (r0, c0) in.  The other axes keep
        // their strides, in order, ahead of it.
        const Stride& in_stride = a->stride();
        const auto s1 = in_stride[static_cast<std::size_t>(a1)];
        const auto s2 = in_stride[static_cast<std::size_t>(a2)];
        Stride stride;
        stride.reserve(out_shape.size());
        for (std::size_t d = 0; d < ndim; ++d)
            if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
                stride.push_back(in_stride[d]);
        stride.push_back(s1 + s2);
        const std::int64_t start = L > 0 ? r0 * s1 + c0 * s2 : 0;
        result = TensorImpl::make_view(a, std::move(out_shape), std::move(stride),
                                       static_cast<std::size_t>(start));
    } else {
        auto out_storage = backend::Dispatcher::for_device(device).diagonal(
            a->storage(), a->shape(), offset, a1, a2, dt);
        result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    }
    auto bwd = std::make_shared<DiagonalBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->offset_ = offset;
    bwd->axis1_ = a1;
    bwd->axis2_ = a2;
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

// ── flip ──────────────────────────────────────────────────────────────────────

// Backward node for flip_op.
//
// Backward formula: flipping is its own inverse — applying the same flip to
// grad_out reproduces the gradient with respect to the input.  No saved
// tensors needed beyond the dims list.
class FlipBackward : public FuncOp<FlipBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<int> dims_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).flip(grad_out, out_shape_, dims_, dtype_)};
    }

    // Graph-mode: flipping is its own inverse, so the derivative is the
    // same rearrangement expressed in ops rather than storages.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {flip_op(grad_out, dims_)};
    }
};

const OpSchema FlipBackward::schema_v1{"flip", 1, AmpPolicy::KeepInput, true};

TensorImplPtr flip_op(const TensorImplPtr& a, std::vector<int> dims) {
    Validator::input(a, "flip").non_null();
    const auto& sh = a->shape();
    const int ndim = static_cast<int>(sh.size());
    dims = distinct_dims(dims, ndim, "flip");
    if (ndim == 0)
        return on_scalar_axis(a, 0, "flip",
                              [](const TensorImplPtr& v, int ax) { return flip_op(v, {ax}); });

    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"flip", device, dt, sh};
    {
        std::vector<std::int64_t> dims_attr(dims.begin(), dims.end());
        scope.set_attr("dims", dims_attr);
    }

    auto& be = backend::Dispatcher::for_device(device);
    Storage out = be.flip(a->storage(), sh, dims, dt);
    auto result = fresh(std::move(out), sh, dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    auto bwd = std::make_shared<FlipBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->dims_ = std::move(dims);
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

// ── masked_select ─────────────────────────────────────────────────────────────

TensorImplPtr masked_select_op(const TensorImplPtr& a, const TensorImplPtr& mask) {
    Validator::input(a, "masked_select.a").non_null();
    Validator::input(mask, "masked_select.mask").non_null();

    auto& be = backend::Dispatcher::for_device(a->device());
    Storage count_s = be.masked_select_count(mask->storage(), mask->shape(), mask->dtype());
    std::int64_t n = 0;
    {
        const auto& cs = std::get<CpuStorage>(count_s);
        std::memcpy(&n, cs.ptr.get(), sizeof(std::int64_t));
    }
    Shape out_shape{n};
    Storage out =
        be.masked_select(a->storage(), mask->storage(), a->shape(), mask->shape(), n, a->dtype());
    return std::make_shared<TensorImpl>(std::move(out), out_shape, a->dtype(), a->device(), false);
}

LUCID_REGISTER_OP(FlipBackward)

}  // namespace lucid
