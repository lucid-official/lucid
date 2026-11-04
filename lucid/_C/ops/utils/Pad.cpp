// lucid/_C/ops/utils/Pad.cpp
//
// Implements constant padding with a differentiable backward pass.
//
// PadBackward undoes the padding by slicing out the original tensor region
// from the incoming gradient.  It iterates over dimensions in order, applying
// one slice_axis call per dimension to reduce the padded shape back to the
// original input shape recorded in input_shapes_[0].
//
// Design note: slicing dimension-by-dimension avoids the need for a
// specialised "multi-dim unpad" kernel in the backend.  Each slice_axis call
// strips the leading pad_width_[d].first elements from the current dimension
// while the narrower target shape implicitly truncates the trailing padding.

#include "Pad.h"

#include <variant>

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
#include "../composite/Indexing.h"  // narrow_op
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;

// Backward node for pad_op.
//
// Invariant: pad_width_[d] holds the (before, after) padding for dimension d
// exactly as provided to the forward call.
//
// Backward formula:
//   For each dimension d, extract the slice [pad_width_[d].first,
//   pad_width_[d].first + input_shapes_[0][d]) from the current gradient
//   buffer.  After all dimensions have been processed the result has exactly
//   the shape of the original input.
class PadBackward : public FuncOp<PadBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::pair<std::int64_t, std::int64_t>> pad_width_;

    std::vector<Storage> apply(Storage grad_out) override {
        auto& be = backend::Dispatcher::for_device(device_);
        Storage current = std::move(grad_out);
        Shape current_shape = out_shape_;
        for (std::size_t d = 0; d < input_shapes_[0].size(); ++d) {
            Shape next_shape = current_shape;
            next_shape[d] = input_shapes_[0][d];
            // Slice away the leading padding; trailing padding is implicitly
            // dropped by the narrower next_shape.
            current = be.slice_axis(current, current_shape, next_shape, static_cast<int>(d),
                                    pad_width_[d].first, dtype_);
            current_shape = std::move(next_shape);
        }
        return {std::move(current)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        // Padding's adjoint is a crop: take back exactly the window the
        // forward wrote the input into, one axis at a time.
        TensorImplPtr current = grad_out;
        for (std::size_t d = 0; d < input_shapes_[0].size(); ++d) {
            current =
                narrow_op(current, static_cast<int>(d), pad_width_[d].first, input_shapes_[0][d]);
        }
        return {current};
    }
};

const OpSchema PadBackward::schema_v1{"pad", 1, AmpPolicy::KeepInput, true};

// Construct a PadBackward node, move the pad_width record into it so that
// the backward can reconstruct the unpadded slice offsets, and wire it onto
// the output tensor.
TensorImplPtr attach_pad_grad(const TensorImplPtr& a,
                              TensorImplPtr out,
                              std::vector<std::pair<std::int64_t, std::int64_t>> pad_width) {
    auto bwd = std::make_shared<PadBackward>();
    bwd->pad_width_ = std::move(pad_width);
    kernel::NaryKernel<PadBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(PadBackward)

}  // namespace

// Validate that pad_width has exactly one entry per dimension.  Compute the
// output shape by adding the before/after widths to each input dimension.
// Dispatch to the backend, which allocates the output buffer, fills it with
// `constant`, and copies the input data into the interior region.  Wire a
// PadBackward node to recover the gradient of the un-padded input.
//
// Critical: the backend must use memset-style initialisation even when
// `constant == 0.0` because pooled allocator buffers may carry stale data.
TensorImplPtr pad_op(const TensorImplPtr& a,
                     std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                     double constant) {
    Validator::input(a, "pad.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"pad", device, dt, a->shape()};
    if (pad_width.size() != a->shape().size())
        ErrorBuilder("pad").invalid_argument("pad_width length must equal ndim");
    const std::size_t ndim = a->shape().size();
    Shape out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        if (pad_width[d].first < 0 || pad_width[d].second < 0)
            ErrorBuilder("pad").invalid_argument("pad_width entries must be non-negative");
        out_shape[d] = a->shape()[d] + pad_width[d].first + pad_width[d].second;
    }
    {
        // Flatten the per-axis (lo, hi) pairs into a single int64 list:
        // [lo_0, hi_0, lo_1, hi_1, ...] — matches MPSGraph's padTensor:
        // expectations (leading/trailing pad arrays).
        std::vector<std::int64_t> pad_attr;
        pad_attr.reserve(2 * pad_width.size());
        for (const auto& p : pad_width) {
            pad_attr.push_back(p.first);
            pad_attr.push_back(p.second);
        }
        scope.set_attr("pads", pad_attr);
        scope.set_attr("constant", constant);
    }

    Storage out_storage = backend::Dispatcher::for_device(device).pad(a->storage(), a->shape(), dt,
                                                                      pad_width, constant);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return attach_pad_grad(a, std::move(result), std::move(pad_width));
}

}  // namespace lucid
