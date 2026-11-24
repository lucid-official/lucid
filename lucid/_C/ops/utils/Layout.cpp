// lucid/_C/ops/utils/Layout.cpp
//
// Implements flatten, broadcast_to, and expand.
//
// flatten delegates entirely to reshape_op: it computes the product of
// shape[start_axis..end_axis], builds a new shape vector with that product
// replacing the flattened range, and forwards to reshape_op.
//
// broadcast_to validates that `a` can be broadcast to the requested shape by
// left-padding with 1s and checking that each dimension is either equal or
// equal to 1 in the padded input.  A strided CPU input is read packed,
// through ``storage()``.  BroadcastBackward reduces the gradient back by
// summing over all axes that were broadcast via Dispatcher::reduce_broadcast.

#include "Layout.h"

#include <algorithm>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
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
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "../ufunc/Reductions.h"  // sum_op
#include "Contiguous.h"
#include "View.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::numel;

}  // namespace

// Collapse dimensions [start_axis, end_axis] (inclusive, after negative-index
// normalisation) into a single dimension whose size is their product.  The
// resulting shape has the dimensions before `s` unchanged, the single flat
// dimension, and then the dimensions after `e`.  Delegates to reshape_op so
// that the autograd node (ViewBackward) is correctly wired.
TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis) {
    Validator::input(a, "flatten.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    const int s = wrap_dim(start_axis, ndim, "flatten");
    const int e = wrap_dim(end_axis, ndim, "flatten");
    // 0-D scalar input flattens to a 1-D length-1 tensor — matches the
    // reference framework's ``tensor(5.0).flatten() == tensor([5.])``
    // contract.
    if (ndim == 0)
        return reshape_op(a, std::vector<std::int64_t>{1});
    if (s > e)
        ErrorBuilder("flatten").invalid_argument("start_dim cannot come after end_dim");

    std::vector<std::int64_t> new_shape;
    for (int d = 0; d < s; ++d)
        new_shape.push_back(a->shape()[d]);
    std::int64_t flat = 1;
    for (int d = s; d <= e; ++d)
        flat *= a->shape()[d];
    new_shape.push_back(flat);
    for (int d = e + 1; d < ndim; ++d)
        new_shape.push_back(a->shape()[d]);
    return reshape_op(a, new_shape);
}

// Schema for BroadcastBackward.  The trailing true marks this op as a potential
// view: the backend may return an aliased (stride-zero) buffer for the forward
// broadcast rather than allocating a physically expanded copy.
const OpSchema BroadcastBackward::schema_v1{"broadcast_to", 1, AmpPolicy::KeepInput, true};

namespace {

// Thin dispatch helper: asks the backend to reduce the broadcast gradient
// over every dimension that was expanded during the forward broadcast.
// `input_shape` and `output_shape` together determine which axes were
// broadcast (any axis where padded input was 1 but output was > 1, and any
// leading axis that was prepended).
Storage reduce_broadcast(const Storage& grad,
                         const Shape& input_shape,
                         const Shape& output_shape,
                         Dtype dt,
                         Device device) {
    return backend::Dispatcher::for_device(device).reduce_broadcast(grad, input_shape, output_shape,
                                                                    dt);
}

}  // namespace

// Sum the output gradient back over every broadcast dimension so its shape
// matches the original input shape stored in input_shape_.  The reduction
// set is computed by the backend from input_shape_ vs. output_shape_.
std::vector<Storage> BroadcastBackward::apply(Storage grad_out) {
    return {reduce_broadcast(grad_out, input_shape_, output_shape_, dtype_, device_)};
}

std::vector<TensorImplPtr> BroadcastBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The adjoint of an expansion is a sum over exactly the axes it
    // invented.  A broadcast left-pads the input shape with 1s, so an axis
    // was invented when the padded input is 1 there and the output is not.
    const std::size_t out_rank = output_shape_.size();
    const std::size_t in_rank = input_shape_.size();
    const std::size_t pad = out_rank - in_rank;

    std::vector<int> axes;
    for (std::size_t i = 0; i < out_rank; ++i) {
        const std::int64_t in_dim = (i < pad) ? 1 : input_shape_[i - pad];
        if (in_dim == 1 && output_shape_[i] != 1)
            axes.push_back(static_cast<int>(i));
    }

    TensorImplPtr reduced = grad_out;
    if (!axes.empty())
        reduced = sum_op(reduced, axes, /*keepdims=*/true);
    // Drop the padded axes and restore the input's own rank.
    std::vector<std::int64_t> target(input_shape_.begin(), input_shape_.end());
    return {reshape_op(reduced, target)};
}

LUCID_REGISTER_OP(BroadcastBackward)

// Left-pad the input's shape with 1s to match the target rank, verify that
// each dimension is either equal or 1 in the padded input, then call
// Dispatcher::broadcast to produce the expanded storage.  Attaches
// BroadcastBackward so that the gradient can be summed back over the
// broadcast dimensions.
//
// Raises ShapeMismatch if:
//   - the input rank is greater than the target rank (cannot broadcast down), or
//   - any dimension is neither equal nor 1 in the padded input shape.
TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape) {
    Validator::input(a, "broadcast_to.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"broadcast_to", device, dt, shape};
    check_concrete_shape(shape);

    // Wrap the broadcast storage, wiring autograd and the tracer to ``a``.
    auto build_with_grad = [&](Storage&& out_storage) {
        auto out = helpers::fresh(std::move(out_storage), shape, dt, device);
        auto bwd = std::make_shared<BroadcastBackward>();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = shape;
        kernel::NaryKernel<BroadcastBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        if (auto* trc = ::lucid::compile::current_tracer()) {
            trc->on_op_io({a}, out);
        }
        return out;
    };

    const std::size_t nin = a->shape().size();
    const std::size_t nout = shape.size();
    if (nin > nout)
        throw ShapeMismatch(shape, a->shape(), "broadcast_to");
    Shape padded(nout, 1);
    std::copy(a->shape().begin(), a->shape().end(), padded.begin() + (nout - nin));
    for (std::size_t d = 0; d < nout; ++d) {
        if (padded[d] != shape[d] && padded[d] != 1)
            throw ShapeMismatch(shape, a->shape(), "broadcast_to");
    }

    Storage out_storage =
        backend::Dispatcher::for_device(device).broadcast(a->storage(), a->shape(), shape, dt);
    return build_with_grad(std::move(out_storage));
}

// expand_op is a direct alias for broadcast_to_op, provided for API symmetry
// with reference framework's Tensor::expand.  Both functions behave identically.
//
// On the CPU ``expand`` is a view: the same buffer, a zero stride on every
// axis it broadcasts.  Reads through it are right — every op reads a strided
// CPU view correctly — and writes are refused, since its elements overlap.
// Metal, and ``broadcast_to`` everywhere, keep copying.
TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape) {
    Validator::input(a, "expand.a").non_null();
    if (!a->views_by_stride())
        return broadcast_to_op(a, shape);
    OpScopeFull scope{"broadcast_to", a->device(), a->dtype(), shape};
    check_concrete_shape(shape);
    const std::size_t nin = a->shape().size();
    const std::size_t nout = shape.size();
    if (nin > nout)
        throw ShapeMismatch(shape, a->shape(), "expand");
    Stride stride(nout, 0);
    for (std::size_t d = nout - nin; d < nout; ++d) {
        const std::size_t src = d - (nout - nin);
        const auto have = a->shape()[src];
        if (have == shape[d])
            stride[d] = a->stride()[src];
        else if (have != 1)
            throw ShapeMismatch(shape, a->shape(), "expand");
    }
    auto out = TensorImpl::make_view(a, shape, std::move(stride), 0);
    auto bwd = std::make_shared<BroadcastBackward>();
    bwd->input_shape_ = a->shape();
    bwd->output_shape_ = shape;
    kernel::NaryKernel<BroadcastBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace lucid
