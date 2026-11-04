// lucid/_C/ops/utils/Tri.cpp
//
// Implements tril_op and triu_op through a shared dispatch helper,
// tri_dispatch, which parameterises upper vs. lower masking.
//
// TriBackward reuses the same tri_storage call as the forward pass: applying
// the same triangular mask to the incoming gradient zeroes gradient elements
// at positions that were zeroed out in the forward output.  This is the
// correct backward because d/dx [x if mask else 0] = 1 if mask else 0,
// i.e. the gradient is passed through wherever the forward was non-zero and
// zeroed wherever the forward was masked to zero.

#include "Tri.h"

#include <variant>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
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
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;

// Thin wrapper that routes to Dispatcher::tri.  The `upper` flag selects
// between tril (upper=false, zeroes above diagonal) and triu (upper=true,
// zeroes below diagonal).  `k` shifts the reference diagonal: 0 is the main
// diagonal, positive values move it toward the upper-right, negative toward
// the lower-left.  The trailing `const char*` parameter is reserved for the
// op name and is intentionally unused here.
Storage tri_storage(const Storage& input,
                    const Shape& shape,
                    Dtype dt,
                    Device device,
                    int k,
                    bool upper,
                    const char*) {
    return backend::Dispatcher::for_device(device).tri(input, shape, dt, k, upper);
}

// Backward node shared by both tril and triu.
//
// Invariants:
//   k_     — diagonal offset passed to the forward call.
//   upper_ — false for tril, true for triu.
//   name_  — human-readable name for profiling/error messages.
//
// Backward formula: the gradient of zero-masking is zero-masking itself.
// Applying the same triangular mask to grad_out zeros gradients that flow
// through the zeroed-out positions in the forward output.
class TriBackward : public FuncOp<TriBackward, 1> {
public:
    static const OpSchema schema_v1;

    int k_ = 0;
    bool upper_ = false;
    const char* name_ = "tril";

    std::vector<Storage> apply(Storage grad_out) override {
        return {tri_storage(grad_out, out_shape_, dtype_, device_, k_, upper_, name_)};
    }

    // Graph-mode: the same mask.  ``tril``/``triu`` zero one triangle, so
    // the gradient is zeroed in exactly the same places — the operation
    // is idempotent and its own derivative.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {upper_ ? triu_op(grad_out, k_) : tril_op(grad_out, k_)};
    }
};

const OpSchema TriBackward::schema_v1{"tri", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

// Wire a TriBackward node onto `out`, recording the diagonal offset `k` and
// the `upper` flag so that the backward can apply the same mask.
TensorImplPtr
attach_tri_grad(const TensorImplPtr& a, TensorImplPtr out, int k, bool upper, const char* name) {
    auto bwd = std::make_shared<TriBackward>();
    bwd->k_ = k;
    bwd->upper_ = upper;
    bwd->name_ = name;
    kernel::NaryKernel<TriBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// Shared forward path for both tril and triu.  Validates that the input is
// non-null, dispatches the triangular masking kernel, creates a fresh output
// TensorImpl, and attaches TriBackward.
//
// `upper` — false for tril, true for triu.
// `name`  — used in error messages and profiler scopes.
TensorImplPtr tri_dispatch(const TensorImplPtr& a, int k, bool upper, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    if (a->shape().size() < 2)
        ErrorBuilder(name).invalid_argument("input must have ndim >= 2");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{name, device, dt, a->shape()};
    scope.set_attr("k", static_cast<std::int64_t>(k));
    scope.set_attr("upper", upper);
    Shape sh = a->shape();
    auto out_storage = tri_storage(a->storage(), sh, dt, device, k, upper, name);
    auto out = fresh(std::move(out_storage), std::move(sh), dt, device);
    return attach_tri_grad(a, std::move(out), k, upper, name);
}

}  // namespace

// Zero elements above diagonal k; pass upper=false to tri_dispatch.
TensorImplPtr tril_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, false, "tril");
}
// Zero elements below diagonal k; pass upper=true to tri_dispatch.
TensorImplPtr triu_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, true, "triu");
}

LUCID_REGISTER_OP(TriBackward)

}  // namespace lucid
