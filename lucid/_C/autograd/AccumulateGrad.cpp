// lucid/_C/autograd/AccumulateGrad.cpp
//
// Implements AccumulateGrad::apply(), which is the final gradient sink for
// every leaf tensor that participates in a backward pass.

#include "AccumulateGrad.h"

#include <utility>

#include "../backend/Dispatcher.h"
#include "../core/Storage.h"
#include "Helpers.h"
#include "TensorHooks.h"

namespace lucid {

AccumulateGrad::AccumulateGrad(std::weak_ptr<TensorImpl> leaf) : leaf_(std::move(leaf)) {}

// Write grad_out into the leaf's gradient storage, or drop it when the leaf
// TensorImpl has been destroyed (weak_ptr expired) — there is nothing to
// accumulate into.  Returns an empty vector because AccumulateGrad has no
// outgoing edges.
std::vector<Storage> AccumulateGrad::apply(Storage grad_out) {
    if (auto t = leaf_.lock())
        commit_leaf_grad(*t, std::move(grad_out));
    return {};
}

// Store grad_out (a TensorImplPtr with its own grad_fn) into the leaf's
// grad_impl slot so the gradient tensor itself is differentiable.
// This path is taken when Engine::backward is called with create_graph=true.
std::vector<TensorImplPtr> AccumulateGrad::apply_for_graph(const TensorImplPtr& grad_out) {
    if (auto t = leaf_.lock())
        commit_leaf_grad_for_graph(t, grad_out);
    return {};
}

bool prepare_leaf_grad(TensorImpl& leaf, Storage& grad_out) {
    if (!leaf.requires_grad())
        return false;
    // The phantom write rule, held before a hook or a cast runs: a real
    // leaf refuses a phantom gradient, its first included.
    if (leaf.absorbs_phantom_gradient(grad_out))
        return false;

    // 3.3 AMP fix: under autocast, the same leaf parameter can be reached
    // via two different effective dtypes — e.g. a Conv with eff_dt=F16
    // emits an F16 grad while a sibling path that ran ForceFP32 emits an
    // F32 grad.  ``accumulate_into`` asserts identical dtype on GPU and
    // would throw DtypeMismatch in that case.  Always cast incoming grads
    // to the leaf parameter's own dtype before storing/accumulating —
    // this matches the reference framework's policy of keeping the
    // gradient slot at the parameter's dtype.
    const Dtype target_dt = leaf.dtype();
    const Dtype src_dt = storage_dtype(grad_out);
    if (src_dt != target_dt) {
        auto& be = backend::Dispatcher::for_device(leaf.device());
        if (is_complex(src_dt) && !is_complex(target_dt)) {
            grad_out = be.complex_real(grad_out, leaf.shape());
            const Dtype lane = real_lane_of(src_dt);
            if (lane != target_dt)
                grad_out = be.astype(grad_out, leaf.shape(), lane, target_dt);
        } else {
            grad_out = be.astype(grad_out, leaf.shape(), src_dt, target_dt);
        }
    }

    grad_out = run_leaf_hooks(leaf, std::move(grad_out));
    return true;
}

void commit_leaf_grad(TensorImpl& leaf, Storage grad_out) {
    if (!leaf.requires_grad())
        return;
    if (leaf.absorbs_phantom_gradient(grad_out))
        return;

    if (leaf.grad_as_impl()) {
        auto g_impl = std::make_shared<TensorImpl>(own_grad_copy(grad_out), leaf.shape(),
                                                   leaf.dtype(), leaf.device(), false);
        leaf.accumulate_grad_impl(std::move(g_impl));
        run_leaf_post_hooks(leaf, /*graph=*/false);
        return;
    }

    auto& grad = leaf.mutable_grad_storage();
    if (!grad.has_value()) {
        // First gradient arriving at this leaf — take ownership directly
        // when uniquely held rather than allocating a zero buffer and adding.
        if (auto* cs = std::get_if<CpuStorage>(&grad_out); cs && cs->ptr.use_count() > 1)
            grad_out = own_grad_copy(grad_out);
        grad = std::move(grad_out);
    } else {
        // Subsequent gradient: add in-place into the existing accumulator.
        accumulate_into(*grad, grad_out);
    }
    run_leaf_post_hooks(leaf, /*graph=*/false);
}

void accumulate_leaf(TensorImpl& leaf, Storage grad_out) {
    if (!prepare_leaf_grad(leaf, grad_out))
        return;
    commit_leaf_grad(leaf, std::move(grad_out));
}

TensorImplPtr prepare_leaf_grad_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad) {
    if (!leaf || !leaf->requires_grad())
        return nullptr;
    if (grad && leaf->absorbs_phantom_gradient(grad->raw_storage()))
        return nullptr;
    return run_leaf_hooks_for_graph(*leaf, gradient_in_dtype_of(grad, leaf));
}

void commit_leaf_grad_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad) {
    if (!leaf || !leaf->requires_grad() || !grad)
        return;
    if (leaf->absorbs_phantom_gradient(grad->raw_storage()))
        return;
    leaf->accumulate_grad_impl(grad);
    run_leaf_post_hooks(*leaf, /*graph=*/true);
}

void accumulate_leaf_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad) {
    TensorImplPtr prepared = prepare_leaf_grad_for_graph(leaf, grad);
    if (!prepared)
        return;
    commit_leaf_grad_for_graph(leaf, prepared);
}

TensorImplPtr gradient_in_dtype_of(const TensorImplPtr& grad, const TensorImplPtr& like) {
    if (!grad || !like || grad->dtype() == like->dtype())
        return grad;
    // The ops layer sits above this one; declared here the way
    // TensorImpl.cpp declares add_op, rather than included.
    extern TensorImplPtr real_op(const TensorImplPtr&);
    extern TensorImplPtr astype_op(const TensorImplPtr&, Dtype);
    TensorImplPtr out = grad;
    if (is_complex(out->dtype()) && !is_complex(like->dtype()))
        out = real_op(out);
    if (out->dtype() != like->dtype())
        out = astype_op(out, like->dtype());
    return out;
}

}  // namespace lucid
