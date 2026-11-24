// lucid/_C/ops/ufunc/Astype.cpp
#include "Astype.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // for lucid::detail::ensure_grad_fn

namespace lucid {

const OpSchema AstypeBackward::schema_v1{"astype", 1, AmpPolicy::KeepInput, true};

// Cast grad_out from dtype_ (= dst_dtype) back to src_dtype_.  Identity
// when the two dtypes match (the forward cast was a no-op).
std::vector<Storage> AstypeBackward::apply(Storage grad_out) {
    if (src_dtype_ == dtype_) {
        return {std::move(grad_out)};
    }
    auto& be = backend::Dispatcher::for_device(device_);
    Storage casted = be.astype(grad_out, out_shape_, dtype_, src_dtype_);
    return {std::move(casted)};
}

// Graph-mode backward: recursive astype_op call so the inverse cast itself
// remains differentiable for higher-order gradients.
std::vector<TensorImplPtr> AstypeBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    if (src_dtype_ == dtype_) {
        return {grad_out};
    }
    return {astype_op(grad_out, src_dtype_)};
}

LUCID_REGISTER_OP(AstypeBackward)

TensorImplPtr astype_op(const TensorImplPtr& a, Dtype dst_dtype) {
    Validator::input(a, "astype").non_null();
    if (a->dtype() == dst_dtype) {
        // Same-dtype no-op: return ``a`` directly so the existing grad_fn
        // chain is preserved.  Prior versions allocated a fresh TensorImpl
        // here, which silently broke autograd routing under AMP because
        // the new wrapper had requires_grad=false.
        return a;
    }
    const auto& shape = a->shape();
    OpScopeFull scope{"astype", a->device(), dst_dtype, shape};
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage out = be.astype(a->storage(), shape, a->dtype(), dst_dtype);
    auto out_impl = helpers::fresh(std::move(out), shape, dst_dtype, a->device());
    // 3.5 Phase 1.3: trace hook — push the destination dtype as an
    // attribute (raw int8 of the Dtype enum) so the MPSGraph emitter
    // can route to the right ``castTensor:toType:`` call.  Pushed
    // unconditionally; eager autograd wiring below still depends on
    // ``GradMode``.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out_impl);
        trc->on_op_attr("dst_dtype", static_cast<std::int64_t>(static_cast<int>(dst_dtype)));
    }

    // Autograd: wire AstypeBackward when the input takes a gradient.  This
    // is the critical hook that makes AMP-driven dtype casts inside
    // Linear / Conv / Matmul / BatchNorm preserve the backward chain.
    // ``maybe_cast_for_amp`` (kernel/BinaryKernel.h) routes through this
    // path; for non-AMP callers (Reductions::argmin → astype to I64, etc.)
    // the input never has requires_grad so the wiring is skipped and the
    // cost stays identical to a raw be.astype().
    // A cast to an integer dtype ends the graph.
    //
    // The derivative of rounding to an integer is zero almost everywhere
    // and undefined on the boundaries, so there is nothing meaningful to
    // propagate — the reference framework refuses the backward outright.
    // Lucid wired one up anyway, and ``x.int().sum().backward()`` did not
    // produce a wrong gradient so much as fall over inside the backend:
    // ``NotImplementedError: cpu_backend::broadcast_back_for_reduce``,
    // several layers below anything the caller could act on.
    //
    // ``float()`` and ``half()`` are unaffected — they stay differentiable
    // and agree with the reference.
    const bool discrete = is_integral(dst_dtype) || dst_dtype == Dtype::Bool;
    if (GradMode::is_enabled() && a->requires_grad() && !discrete) {
        auto bwd = std::make_shared<AstypeBackward>();
        bwd->src_dtype_ = a->dtype();
        bwd->dtype_ = dst_dtype;
        bwd->device_ = a->device();
        bwd->input_shapes_[0] = shape;
        bwd->out_shape_ = shape;
        bwd->input_tensors_[0] = a;
        bwd->saved_impl_inputs_[0] = a;

        std::vector<Edge> edges;
        edges.emplace_back(lucid::detail::ensure_grad_fn(a), a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});

        out_impl->set_grad_fn(std::move(bwd));
        out_impl->set_leaf(false);
        out_impl->set_requires_grad(true);
    }
    return out_impl;
}

}  // namespace lucid
