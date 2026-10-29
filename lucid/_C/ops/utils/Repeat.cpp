// lucid/_C/ops/utils/Repeat.cpp
//
// Implements repeat_op and tile_op together with their autograd nodes.
//
// RepeatBackward: reduces the tiled gradient along a single axis by summing
//   every `repeats_`-sized block back to a single value, effectively undoing
//   the replication.  Delegates to Dispatcher::repeat_backward.
//
// TileBackward: reduces the tiled gradient over all tiled dimensions by
//   summing each set of repeated contributions.  Delegates to
//   Dispatcher::tile_backward, which is given the pre-computed padded input
//   shape so it can correctly attribute each gradient cell to the right
//   source element in the original (possibly left-padded) input.

#include "Repeat.h"

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "../ufunc/Reductions.h"
#include "View.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::wrap_axis;

// Backward node for repeat_op.
//
// Invariants:
//   axis_    — the axis along which the forward pass repeated elements.
//   repeats_ — the repetition count used in the forward pass.
//
// Backward formula: sum every `repeats_`-sized block along axis_ to recover
// the gradient w.r.t. the original input.
class RepeatBackward : public FuncOp<RepeatBackward, 1> {
public:
    static const OpSchema schema_v1;

    int axis_ = 0;
    std::int64_t repeats_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).repeat_backward(
            grad_out, input_shapes_[0], out_shape_, axis_, repeats_, dtype_)};
    }

    // The same sum, recorded: split the repeated axis into (n, repeats) and
    // sum the copies away.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const Shape& in = input_shapes_[0];
        const int nd = static_cast<int>(in.size());
        const int axis = axis_ < 0 ? axis_ + nd : axis_;
        std::vector<std::int64_t> split(in.begin(), in.end());
        split.insert(split.begin() + axis + 1, repeats_);
        return {sum_op(reshape_op(grad_out, split), {axis + 1}, false)};
    }
};

const OpSchema RepeatBackward::schema_v1{"repeat", 1, AmpPolicy::KeepInput, true};

// Backward node for tile_op.
//
// Invariants:
//   padded_shape_ — the input shape after left-padding with 1s to match the
//                   length of reps_.  Needed by tile_backward to compute
//                   per-dimension reduction factors.
//   reps_         — the repetition counts per dimension from the forward call.
//
// Backward formula: for each dimension d, sum the reps_[d] gradient tiles
// back into a region of size padded_shape_[d].
class TileBackward : public FuncOp<TileBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape padded_shape_;
    std::vector<std::int64_t> reps_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).tile_backward(
            grad_out, input_shapes_[0], padded_shape_, out_shape_, reps_, dtype_)};
    }

    // The same sum, recorded: each tiled axis of length reps * size splits
    // into (reps, size) — copy t of element j sits at t * size + j — and the
    // copies are summed away.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        std::vector<std::int64_t> split;
        std::vector<int> copies;
        for (std::size_t d = 0; d < reps_.size(); ++d) {
            copies.push_back(static_cast<int>(split.size()));
            split.push_back(reps_[d]);
            split.push_back(padded_shape_[d]);
        }
        auto summed = sum_op(reshape_op(grad_out, split), copies, false);
        const Shape& in = input_shapes_[0];
        return {reshape_op(summed, std::vector<std::int64_t>(in.begin(), in.end()))};
    }
};

const OpSchema TileBackward::schema_v1{"tile", 1, AmpPolicy::KeepInput, true};

// Construct a RepeatBackward node, record the axis and repeat count from the
// forward call, then wire the node onto the output tensor.
TensorImplPtr
attach_repeat_grad(const TensorImplPtr& a, TensorImplPtr out, int axis, std::int64_t repeats) {
    auto bwd = std::make_shared<RepeatBackward>();
    bwd->axis_ = axis;
    bwd->repeats_ = repeats;
    kernel::NaryKernel<RepeatBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// Construct a TileBackward node, recording the padded input shape and the
// repetition vector.  The padded shape is needed by tile_backward to resolve
// per-dimension reduction factors (reps_[d] tiles per padded dimension d).
TensorImplPtr attach_tile_grad(const TensorImplPtr& a,
                               TensorImplPtr out,
                               Shape padded_shape,
                               std::vector<std::int64_t> reps) {
    auto bwd = std::make_shared<TileBackward>();
    bwd->padded_shape_ = std::move(padded_shape);
    bwd->reps_ = std::move(reps);
    kernel::NaryKernel<TileBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(RepeatBackward)
LUCID_REGISTER_OP(TileBackward)

}  // namespace

// Tile `a` along `axis` exactly `repeats` times.  The output shape is
// identical to the input shape except that dimension `ax` is multiplied by
// `repeats`.  Attaches RepeatBackward so gradients can be summed back.
TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis) {
    Validator::input(a, "repeat.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"repeat", device, dt, a->shape()};
    // Before the backend sizes its copy loop by it: the reference refuses a
    // negative count even when the axis is empty.
    if (repeats < 0)
        ErrorBuilder("repeat").invalid_argument("Repeats must be non-negative, got " +
                                                std::to_string(repeats));
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()), "repeat");
    scope.set_attr("axis", static_cast<std::int64_t>(ax));
    scope.set_attr("repeats", static_cast<std::int64_t>(repeats));
    Shape out_shape = a->shape();
    out_shape[ax] = mul_extent(out_shape[ax], repeats);
    check_concrete_shape(out_shape);
    Storage out_storage =
        backend::Dispatcher::for_device(device).repeat(a->storage(), a->shape(), dt, repeats, ax);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return attach_repeat_grad(a, std::move(result), ax, repeats);
}

// Tile `a` according to `reps`, one repetition count per output dimension.
// When reps.size() > a.ndim the input is conceptually left-padded with size-1
// dimensions; `padded` captures the extended shape (with the leading 1s) and
// is saved into TileBackward so it can correctly map gradient regions to their
// source positions in the original (unpadded) input.
TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps) {
    Validator::input(a, "tile.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"tile", device, dt, a->shape()};
    const std::size_t nout = reps.size();
    if (nout < a->shape().size())
        ErrorBuilder("tile").invalid_argument("reps must be at least as long as ndim");

    // Build padded: align the input dimensions against the right of reps by
    // prepending 1s for every extra leading repetition dimension.
    Shape padded(nout, 1);
    const std::size_t lead = nout - a->shape().size();
    for (std::size_t d = 0; d < a->shape().size(); ++d)
        padded[lead + d] = a->shape()[d];
    // The reference rules on the output's extents, not the counts: a negative
    // count over an empty axis still makes an empty one.  The backends size
    // their loops by the count, so they are handed 0 there, and any other
    // negative extent is refused before they run.
    Shape out_shape(nout);
    for (std::size_t d = 0; d < nout; ++d) {
        if (padded[d] == 0 && reps[d] < 0)
            reps[d] = 0;
        out_shape[d] = mul_extent(padded[d], reps[d]);
    }
    check_concrete_shape(out_shape);
    scope.set_attr("reps", reps);

    Storage out_storage =
        backend::Dispatcher::for_device(device).tile(a->storage(), a->shape(), dt, reps);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return attach_tile_grad(a, std::move(result), std::move(padded), std::move(reps));
}

}  // namespace lucid
