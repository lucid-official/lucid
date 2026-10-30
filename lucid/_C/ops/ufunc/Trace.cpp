// lucid/_C/ops/ufunc/Trace.cpp
//
// Trace forward (sum of diagonal elements) and backward (scatter gradient
// to diagonal positions).
//
// Backward: IBackend::trace_backward writes the scalar gradient to each
// diagonal element of a zero-filled matrix with the same shape as the input.
// For a [m, n] input the backward output is a [m, n] matrix that is zero
// everywhere except at positions (i, i).
//
// Autograd is only wired for 2-D inputs; batch trace (ndim > 2) returns a
// plain tensor without a backward node.
//
// TraceBackward is in an anonymous namespace because its interface is internal.

#include "Trace.h"

#include <algorithm>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Mul.h"
#include "../bfunc/_BinaryOp.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Layout.h"
#include "Astype.h"
#include "Reductions.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// Private backward node for trace.
//
// Saved state:
//   input_shape_ — shape of the forward input; the backend needs it to
//                  reconstruct the diagonal-scatter output.
class TraceBackward : public AutogradNode<TraceBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;

    // Delegates entirely to IBackend::trace_backward, which scatters
    // grad_out to the diagonal positions of a zero matrix.
    std::vector<Storage> apply(Storage grad_out) override {
        return {backend::Dispatcher::for_device(device_).trace_backward(grad_out, input_shape_,
                                                                        dtype_)};
    }

    // g on the diagonal, zero elsewhere — g times the identity — recorded.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const std::int64_t rows = input_shape_[0];
        const std::int64_t cols = input_shape_[1];
        return {mul_op(broadcast_to_op(grad_out, input_shape_),
                       eye_op(rows, cols, 0, dtype_, device_))};
    }
};

// KeepInput: trace is valid for integer and float dtypes alike.
const OpSchema TraceBackward::schema_v1{"trace", 1, AmpPolicy::KeepInput, true};

}  // namespace

// Dispatch the trace forward, then wire TraceBackward for 2-D inputs.
// For ndim > 2 the last two dimensions are contracted; the output shape is
// the batch prefix (sh[2:]).  Autograd is currently omitted for the batch case
// because trace_backward only supports 2-D inputs.
TensorImplPtr trace_op(const TensorImplPtr& a_in) {
    Validator::input(a_in, "trace.a").non_null();
    if (a_in->shape().size() < 2)
        ErrorBuilder("trace").fail("input must have ndim >= 2");
    // The narrow integers add up in int64, as ``sum`` does.  Bool stays as it
    // is: the backend refuses it, as the reference refuses ``trace`` of a bool.
    const TensorImplPtr wide = a_in->dtype() == Dtype::Bool ? a_in : promote_int_for_reduce(a_in);
    SchemaGuard sg{TraceBackward::schema_v1, wide->dtype(), wide->device()};
    const TensorImplPtr a = astype_op(wide, sg.effective_dtype());
    const Dtype dt = a->dtype();
    const Device device = a->device();
    const auto& sh = a->shape();
    OpScopeFull scope{"trace", device, dt, Shape{}};

    // Output shape: drop the first two (matrix) dimensions; keep the batch prefix.
    Shape out_shape(sh.begin() + 2, sh.end());
    auto out_storage = backend::Dispatcher::for_device(device).trace(a->storage(), a->shape(), dt);
    TensorImplPtr out = fresh(std::move(out_storage), out_shape, dt, device);

    // Wire autograd only for the 2-D case; batch trace does not propagate gradients.
    if (a->shape().size() == 2) {
        auto bwd = std::make_shared<TraceBackward>();
        bwd->input_shape_ = a->shape();
        kernel::NaryKernel<TraceBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

}  // namespace lucid
