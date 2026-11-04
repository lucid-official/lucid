// lucid/_C/ops/ufunc/ToDevice.cpp
#include "ToDevice.h"

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/PhantomMode.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // for lucid::detail::ensure_grad_fn

namespace lucid {

const OpSchema ToDeviceBackward::schema_v1{"to_device", 1, AmpPolicy::KeepInput, true};

// Gradients arrive as fresh, contiguous buffers of the output's shape on
// the output's device, which is all the upload and download paths need.
std::vector<Storage> ToDeviceBackward::apply(Storage grad_out) {
    if (src_device_ == device_)
        return {std::move(grad_out)};
    if (storage_is_phantom(grad_out))
        return {phantom_storage(out_shape_, dtype_, src_device_)};
    if (src_device_ == Device::GPU)
        return {Storage{gpu::upload_cpu_to_gpu(storage_cpu(grad_out), out_shape_)}};
    return {Storage{gpu::download_gpu_to_cpu(storage_gpu(grad_out), out_shape_)}};
}

std::vector<TensorImplPtr> ToDeviceBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {to_device_op(grad_out, src_device_)};
}

LUCID_REGISTER_OP(ToDeviceBackward)

TensorImplPtr to_device_op(const TensorImplPtr& a, Device target) {
    Validator::input(a, "to_device").non_null();
    if (a->device() == target)
        return a;

    // A shared buffer is relabelled, not copied — see TensorImpl::metal_shared.
    TensorImplPtr out;
    if (const SharedStorage* sh = a->metal_shared())
        out = std::make_shared<TensorImpl>(Storage{*sh}, a->shape(), a->dtype(), target, false);
    else
        out = a->transfer_to_device(target, false);

    // A copy of a traced value is recorded as a ``to_device`` op.  The trace
    // runs on one device, so a compiled call holding a move runs eagerly; an
    // unrecorded copy was a tensor the trace had never seen, and an op or
    // host read of it kept the first call's values for every later call.
    // A move of a constant — a tensor the trace has not seen — records
    // nothing (Tracer::on_derived).
    compile::traced_derive("to_device", a, out);

    // No saved version for ``a``: the derivative of a move never reads its
    // values, so writing to ``a`` in place after the move is not an error.
    if (GradMode::is_enabled() && a->requires_grad()) {
        auto bwd = std::make_shared<ToDeviceBackward>();
        bwd->src_device_ = a->device();
        bwd->dtype_ = a->dtype();
        bwd->device_ = target;
        bwd->input_shapes_[0] = a->shape();
        bwd->out_shape_ = a->shape();
        bwd->input_tensors_[0] = a;

        std::vector<Edge> edges;
        edges.emplace_back(lucid::detail::ensure_grad_fn(a), a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
    }
    return out;
}

}  // namespace lucid
