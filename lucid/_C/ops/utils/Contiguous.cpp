// lucid/_C/ops/utils/Contiguous.cpp
//
// Implements the contiguous op, which guarantees that the output tensor uses
// a dense, offset-free row-major (C-order) storage layout.  The backend
// dispatcher examines the is_contiguous flag together with the stride/offset
// metadata to decide whether a copy is actually needed.
//
// Design note: the forward static method on ContiguousBackward combines the
// forward compute step and the autograd wiring into a single call to keep the
// public contiguous_op entry point trivially thin.

#include "Contiguous.h"

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
#include "../bfunc/_BinaryOp.h"

namespace lucid {

// Schema for ContiguousBackward; the fourth field (true) marks this op as a
// potential view (the forward may return the same storage unchanged when the
// input is already contiguous).
const OpSchema ContiguousBackward::schema_v1{"contiguous", 1, AmpPolicy::KeepInput, true};

// Allocate (or reuse) a contiguous buffer for `a` and attach the backward
// node.  Passes stride, storage_offset, and is_contiguous to the dispatcher
// so the backend can skip a physical copy when the data is already dense.
TensorImplPtr ContiguousBackward::forward(const TensorImplPtr& a) {
    Validator::input(a, "contiguous.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};

    // The CPU backend walks the buffer through the geometry; the Metal one
    // ignores the geometry and copies what it is handed, so it is handed the
    // tensor's own elements — a family member's root holds more than them.
    // A phantom is read through storage() as any op input is, which refuses
    // it outside phantom mode.
    const Storage& src = storage_is_cpu(a->raw_storage()) ? a->raw_storage() : a->storage();
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .contiguous(src, a->shape(), a->stride(), a->storage_offset(),
                                          a->is_contiguous(), a->dtype());

    auto result = helpers::fresh(std::move(out_storage), a->shape(), a->dtype(), a->device());

    kernel::NaryKernel<ContiguousBackward, 1>::wire_autograd({a}, result, false);
    return result;
}

// The gradient of making a tensor contiguous is the identity: the incoming
// gradient already matches the output shape and dtype, so a plain clone is
// sufficient to give the upstream node a concrete, owning buffer.
std::vector<Storage> ContiguousBackward::apply(Storage grad_out) {
    return {clone_storage(grad_out, shape_numel(out_shape_), dtype_, device_)};
}

TensorImplPtr contiguous_op(const TensorImplPtr& a) {
    return ContiguousBackward::forward(a);
}

LUCID_REGISTER_OP(ContiguousBackward)

}  // namespace lucid
