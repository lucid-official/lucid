// lucid/_C/ops/utils/Meshgrid.cpp
//
// Implements meshgrid_op with autograd support via MeshgridBackward.
//
// Forward: delegates grid construction to Dispatcher::meshgrid, which
// broadcasts each 1-D input across the full N-dimensional output shape.
// For "xy" indexing (indexing_xy=true) the first two output dimensions are
// swapped so that input[0] (x) varies along axis 1 and input[1] (y) along
// axis 0; all remaining inputs follow the "ij" convention (input[i] along i).
//
// Backward: MeshgridBackward recovers the 1-D gradient for each input by
// summing the N-D output gradient over every axis except the carry axis.
// For "xy" indexing, the carry axes for the first two inputs are swapped
// (carry_axis_=1 for input[0], carry_axis_=0 for input[1]) to match how the
// forward pass assigned them.

#include "Meshgrid.h"

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "../ufunc/Reductions.h"
#include "View.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::check_dtype_device_match;
using utils_detail::fresh;

// Reduce the N-D output gradient to a 1-D gradient for one meshgrid input.
//
// Builds an axis list containing every dimension except `carry_axis`, then
// sums the gradient over those axes.  The result has the same shape as the
// original 1-D input (size = output_shape[carry_axis]).
//
// Special case: if N=1 there are no axes to sum over and a plain clone of
// the gradient is returned to avoid a degenerate empty-axis reduction.
Storage meshgrid_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int carry_axis,
                                  Dtype dt,
                                  Device device) {
    std::vector<int> axes;
    axes.reserve(output_shape.size() > 0 ? output_shape.size() - 1 : 0);
    for (int d = 0; d < static_cast<int>(output_shape.size()); ++d) {
        if (d != carry_axis)
            axes.push_back(d);
    }
    auto& backend = backend::Dispatcher::for_device(device);
    if (axes.empty())
        return backend.clone(grad, input_shape, dt);
    return backend.reduce_sum(grad, output_shape, backend::ReduceOpts{axes, false}, dt);
}

// Backward node for a single meshgrid output tensor.
//
// Invariant:
//   carry_axis_ — the axis in the output grid along which the corresponding
//                 1-D input varies.  For "ij" indexing, carry_axis_ == i.
//                 For "xy" indexing, inputs 0 and 1 have their carry axes
//                 swapped (carry_axis_ == 1 for input 0 and 0 for input 1).
//
// Backward formula: sum grad_out over all axes except carry_axis_ to
// collapse the broadcast dimensions back to a 1-D shape.
class MeshgridBackward : public FuncOp<MeshgridBackward, 1> {
public:
    static const OpSchema schema_v1;

    int carry_axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        return {meshgrid_backward_storage(grad_out, input_shapes_[0], out_shape_, carry_axis_,
                                          dtype_, device_)};
    }

    // The same sum over every axis the input was spread along, recorded.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        std::vector<int> spread;
        for (int d = 0; d < static_cast<int>(out_shape_.size()); ++d)
            if (d != carry_axis_)
                spread.push_back(d);
        auto summed = spread.empty() ? grad_out : sum_op(grad_out, spread, false);
        const Shape& in = input_shapes_[0];
        return {reshape_op(summed, std::vector<std::int64_t>(in.begin(), in.end()))};
    }
};

const OpSchema MeshgridBackward::schema_v1{"meshgrid", 1, AmpPolicy::KeepInput, true};

// Wire a MeshgridBackward node for one output grid.
TensorImplPtr
attach_meshgrid_grad(const TensorImplPtr& input, TensorImplPtr output, int carry_axis) {
    auto bwd = std::make_shared<MeshgridBackward>();
    bwd->carry_axis_ = carry_axis;
    kernel::NaryKernel<MeshgridBackward, 1>::wire_autograd(std::move(bwd), {input}, output, false);
    return output;
}

}  // namespace

// Build the N-dimensional coordinate grids.  Each output[i] is an
// N-dimensional tensor where dimension carry_axis varies and all other
// dimensions are size 1 broadcast to the full output shape.
std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs, bool indexing_xy) {
    check_dtype_device_match(xs, "meshgrid");
    const Dtype dt = xs[0]->dtype();
    const Device device = xs[0]->device();
    OpScopeFull scope{"meshgrid", device, dt, Shape{}};
    // Which axis each input varies along is the whole difference between
    // the two indexings, and the output shapes only reveal it when the
    // inputs have different lengths.  Recorded so a consumer of the
    // trace does not have to guess.
    scope.set_attr("indexing_xy", static_cast<std::int64_t>(indexing_xy ? 1 : 0));
    const std::size_t N = xs.size();
    std::vector<std::int64_t> dims(N);
    for (std::size_t i = 0; i < N; ++i) {
        if (xs[i]->shape().size() > 1)
            ErrorBuilder("meshgrid").fail("each input must be 1-D");
        dims[i] = xs[i]->shape().empty() ? 1 : xs[i]->shape()[0];
    }
    // For "xy" indexing swap the first two output dimensions so that the x
    // (first) input varies along axis 1 and the y (second) input along axis 0.
    std::vector<std::int64_t> out_dims = dims;
    if (indexing_xy && N >= 2)
        std::swap(out_dims[0], out_dims[1]);
    Shape out_shape(out_dims.begin(), out_dims.end());
    std::vector<Storage> in_storage;
    in_storage.reserve(xs.size());
    for (const auto& x : xs)
        in_storage.push_back(x->storage());
    std::vector<Storage> out_storage =
        backend::Dispatcher::for_device(device).meshgrid(in_storage, out_shape, dt, indexing_xy);

    std::vector<TensorImplPtr> result;
    result.reserve(out_storage.size());
    for (std::size_t i = 0; i < out_storage.size(); ++i) {
        // For "xy" indexing the first two carry axes are swapped: input 0 (x)
        // must vary along axis 1 (columns) and input 1 (y) along axis 0 (rows).
        // The expression `1 - i` maps i=0 → 1 and i=1 → 0, achieving the swap
        // without a conditional for each case.
        std::size_t carry_axis = i;
        if (indexing_xy && N >= 2 && i < 2)
            carry_axis = 1 - i;
        auto out = fresh(std::move(out_storage[i]), out_shape, dt, device);
        result.push_back(attach_meshgrid_grad(xs[i], std::move(out), static_cast<int>(carry_axis)));
    }

    // Each output was wired for autograd against its own single input,
    // and ``on_op_io`` takes the last write — so the node ended up
    // recording one input for however many it was given, and a consumer
    // could not tell which. Recorded once more here with the full list:
    // a non-empty input list replaces the node's inputs, and naming an
    // output that is already registered only re-confirms its id.
    if (auto* trc = ::lucid::compile::current_tracer(); trc && !result.empty())
        trc->on_op_io(xs, result[0]);
    return result;
}

LUCID_REGISTER_OP(MeshgridBackward)

}  // namespace lucid
