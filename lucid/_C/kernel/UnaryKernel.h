// lucid/_C/kernel/UnaryKernel.h
//
// CRTP base for single-input, single-output op kernels — pairs a typed
// ``forward()`` trampoline with the one-input saved-tensor bookkeeping
// every unary op needs.
//
// Concrete unary ops (``NegBackward``, ``LogBackward``, ``ReluBackward``,
// :class:`Abs`, :class:`Sin`, …) inherit as
// ``class FooBackward : public UnaryKernel<FooBackward>`` and supply the
// op-specific schema and compute hooks; the base owns dtype promotion,
// contiguity enforcement, backend dispatch, and full autograd wiring so
// concrete ops only have to write the math.
//
// Forward dispatch priority
// -------------------------
// 1. ``Derived::dispatch(IBackend&, ...)`` — preferred when the op has
//    a fused backend-aware implementation (matched by
//    :type:`detail::HasUnaryDispatch`).
// 2. ``Derived::gpu_kernel(GpuStorage, ...)`` — chosen on GPU when the
//    op exposes a typed GPU kernel (matched by
//    :type:`detail::HasUnaryGpuKernel`).
// 3. ``Derived::cpu_kernel(CpuStorage, ...)`` — fallback CPU path.
//
// The :meth:`apply` override delegates to ``Derived::grad_formula(grad_out)``
// and broadcast-reduces the result back to the original input shape;
// :meth:`apply_for_graph` provides the create_graph=True higher-order
// path via ``Derived::grad_formula_impl``.

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../compile/Tracer.h"  // 3.5 Phase 1.2 step 2: trace I/O wiring at forward
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpSchema.h"
#include "../core/PhantomMode.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "Contig.h"
#include "IKernel.h"

namespace lucid {

namespace detail {

// Concept satisfied when ``Derived`` exposes a typed unary GPU kernel.
//
// Template Parameters
// -------------------
// T : class
//     The candidate Derived kernel class.
//
// Notes
// -----
// Matches the call ``T::gpu_kernel(GpuStorage, Shape, Dtype) -> GpuStorage``.
// :class:`UnaryKernel` uses this concept to enable the typed GPU path
// only for ops that implement it; ops without a GPU kernel fall through
// to a clear ``not_implemented`` error.
template <class T>
concept HasUnaryGpuKernel = requires(GpuStorage a, Shape s, Dtype d) {
    { T::gpu_kernel(a, s, d) } -> std::same_as<GpuStorage>;
};

// Concept satisfied when ``Derived`` opts in to backend-agnostic dispatch.
//
// Template Parameters
// -------------------
// T : class
//     The candidate Derived kernel class.
//
// Notes
// -----
// Matches the call
// ``T::dispatch(IBackend&, Storage, Shape, Dtype) -> Storage``.  Ops
// that need to route through the :class:`backend::IBackend` abstraction
// (rather than typed CPU/GPU kernels) provide this; the trampoline in
// :meth:`UnaryKernel::forward` prefers ``dispatch`` over the typed
// kernel paths when both are present.
template <class T>
concept HasUnaryDispatch = requires(backend::IBackend& be, Storage a, Shape s, Dtype d) {
    { T::dispatch(be, a, s, d) } -> std::same_as<Storage>;
};

}  // namespace detail

// CRTP base for single-input, single-output op kernels.
//
// Inherits :class:`AutogradNode\<Derived, 1\>` (one saved input slot)
// and :class:`kernel::IKernel` so an instance can be held polymorphically
// while still exposing the typed static ``forward()`` trampoline.
//
// Concrete ops declare themselves as
// ``class FooBackward : public UnaryKernel<FooBackward>`` and provide:
//
//   - ``static constexpr OpSchema schema_v1`` — op name + AMP policy.
//   - ``static cpu_kernel(a, out_shape, dtype) -> CpuStorage`` and/or
//     ``static gpu_kernel(a, out_shape, dtype) -> GpuStorage``, **or**
//     ``static dispatch(IBackend&, a, out_shape, dtype) -> Storage``.
//   - ``Storage grad_formula(Storage grad_out)`` — the local Jacobian
//     product computed at backward time.
//   - Optional ``grad_formula_impl(grad_out, a_impl, out_impl) -> TensorImplPtr``
//     for graph-mode (create_graph=True) higher-order differentiation.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete CRTP self-type.
//
// Attributes
// ----------
// kSavesInput : static constexpr bool
//     Whether ``forward()`` snapshots ``a->storage()`` into
//     ``saved_inputs_[0]`` for use in ``grad_formula``.  Defaults to
//     ``true``; set ``false`` when the gradient formula uses only the
//     saved output (e.g. :class:`ReLU` keyed on output sign).
// kSavesOutput : static constexpr bool
//     Whether ``forward()`` additionally snapshots the output storage
//     into ``saved_output_``.  Defaults to ``false``.  Set ``true`` for
//     ops whose backward is cheap in terms of ``y`` (e.g. :class:`Exp`,
//     :class:`Sigmoid`).
// kHasGradient : static constexpr bool
//     Whether the op participates in autograd.  Defaults to ``true``;
//     set ``false`` for non-differentiable ops to skip all graph wiring.
//
// Notes
// -----
// AMP policy and dtype promotion are forwarded through
// :class:`SchemaGuard` using ``Derived::schema_v1``; the effective dtype
// returned drives both the input cast and the saved-input dtype.
//
// See Also
// --------
// :class:`BinaryKernel`, :class:`NaryKernel`, :class:`VariadicKernel`.
// :class:`IKernel` — the abstract base above the CRTP layer.
template <class Derived>
class UnaryKernel : public AutogradNode<Derived, 1>, public kernel::IKernel {
public:
    // Snapshot the forward input storage into ``saved_inputs_[0]``.
    //
    // Defaults to ``true``.  Set ``false`` in concrete ops whose
    // backward formula does not need the input — this elides the
    // per-forward storage copy.
    static constexpr bool kSavesInput = true;

    // Snapshot the forward output storage into ``saved_output_``.
    //
    // Defaults to ``false``.  Enable when the gradient is more cheaply
    // expressed in terms of the output (e.g. :class:`Exp` whose
    // derivative equals ``y`` itself).
    static constexpr bool kSavesOutput = false;

    // Whether the op participates in autograd at all.
    //
    // Defaults to ``true``.  Set ``false`` for non-differentiable
    // ops (integer cast, copy, in-place mutations) so ``forward()``
    // skips graph wiring entirely.
    static constexpr bool kHasGradient = true;

    // Default graph-mode gradient formula — concrete ops override.
    //
    // Parameters
    // ----------
    // g : const TensorImplPtr&
    //     Gradient of the loss with respect to this op's output, as a
    //     :class:`TensorImpl` that itself carries ``grad_fn`` so the
    //     backward computation is differentiable.
    // a : const TensorImplPtr&
    //     The saved forward input (possibly broadcast / cast to the
    //     effective dtype).
    // out : const TensorImplPtr&
    //     The saved forward output (populated only when
    //     ``kSavesOutput == true``).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     ``grad_input`` as a fully-traced :class:`TensorImpl`.
    //
    // Raises
    // ------
    // std::runtime_error
    //     The default base implementation always throws.  Concrete ops
    //     must override to support ``create_graph=True``.
    TensorImplPtr grad_formula_impl(const TensorImplPtr& /*g*/,
                                    const TensorImplPtr& /*a*/,
                                    const TensorImplPtr& /*out*/) {
        throw std::runtime_error("create_graph=True is not supported for op '" +
                                 std::string(Derived::schema_v1.name) +
                                 "'. "
                                 "Implement grad_formula_impl() to add support.");
    }

    // Return the canonical schema name of the concrete op.
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Return the autograd node label (same string as :meth:`name`).
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Typed forward trampoline for a single-input op.
    //
    // Parameters
    // ----------
    // a : const std::shared_ptr<TensorImpl>&
    //     The input tensor.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     The output tensor.  If ``a`` requires a gradient (and grad
    //     mode is enabled) the result has its ``grad_fn`` set to a
    //     freshly constructed ``Derived`` backward node.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the schema's shape contract is violated.
    // DtypeMismatch
    //     If the input dtype is not compatible with the schema's
    //     accepted dtype set.
    //
    // Notes
    // -----
    // Behaviour
    //
    //   1. Validate ``a`` is non-null.
    //   2. Resolve the effective dtype via :class:`SchemaGuard`.
    //   3. Materialise contiguous on CPU when ``a`` is non-contiguous.
    //   4. Cast to the effective dtype if needed.
    //   5. Dispatch to ``dispatch`` / ``gpu_kernel`` / ``cpu_kernel``
    //      (see file header for priority).
    //   6. Wire the autograd graph when ``kHasGradient`` and any input
    //      requires a gradient.
    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a);

    // Backward implementation invoked by the autograd engine.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     The upstream gradient with respect to this op's output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element vector containing ``grad_input`` reduced back
    //     to ``input_shapes_[0]`` via :func:`reduce_grad_to_shape`.
    //
    // Notes
    // -----
    // The reduction step is needed when the forward broadcast the
    // input to a larger output shape; for purely shape-preserving
    // unary ops it is a no-op.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward — supports create_graph=True.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     The upstream gradient, retained as a :class:`TensorImpl` with
    //     its own ``grad_fn`` for higher-order differentiation.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     Single-element vector containing the fully-traced
    //     ``grad_input`` reduced back to the original input shape via
    //     :func:`sum_op` / :func:`reshape_op`.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If the saved input pointer was not captured (i.e. the forward
    //     ran without create_graph=True), or if the concrete op did not
    //     override :meth:`grad_formula_impl`.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

protected:
    // The result's storage at ``a_ptr``'s shape and ``eff_dt``: the
    // backend's dispatch, or the Derived CPU / GPU kernel.  A phantom under
    // phantom mode, where no kernel runs.
    static Storage compute_storage(const TensorImplPtr& a_ptr, Dtype eff_dt);
};

template <class Derived>
Storage UnaryKernel<Derived>::compute_storage(const TensorImplPtr& a_ptr, Dtype eff_dt) {
    if (PhantomMode::is_enabled()) [[unlikely]]
        return phantom_storage(a_ptr->shape(), eff_dt, a_ptr->device());
    if constexpr (detail::HasUnaryDispatch<Derived>) {
        // Dispatch path: uses the backend abstraction (Accelerate / MLX).
        return Derived::dispatch(backend::Dispatcher::for_device(a_ptr->device()), a_ptr->storage(),
                                 a_ptr->shape(), eff_dt);
    } else {
        if (a_ptr->device() == Device::GPU) {
            if constexpr (detail::HasUnaryGpuKernel<Derived>) {
                return Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()),
                                                   a_ptr->shape(), eff_dt)};
            } else {
                ErrorBuilder(Derived::schema_v1.name)
                    .not_implemented("GPU kernel not yet implemented");
            }
        }
        return Storage{
            Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()), a_ptr->shape(), eff_dt)};
    }
}

// Out-of-class definition of :meth:`UnaryKernel::forward`.
//
// See the in-class declaration for parameter and return semantics.
template <class Derived>
std::shared_ptr<TensorImpl> UnaryKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    // SchemaGuard resolves the effective dtype (e.g., promoting F16 → F32
    // when the schema requires full precision) and validates device support.
    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    // A strided CPU input needs no copy here: ``storage()`` hands the kernel
    // its elements packed.  Calling ``contiguous`` inside this op's scope put
    // a second op between this one and its real input, which the compile and
    // Core ML tracers could not follow.
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a, eff_dt);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, a_ptr->shape()};

    Storage out_storage = compute_storage(a_ptr, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a_ptr->shape(), eff_dt,
                                            a_ptr->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    // 3.5 Phase 1.2 step 2: trace I/O wiring at the forward boundary.
    // UnaryKernel doesn't go through NaryKernel::wire_autograd, so the
    // hook here keeps element-wise + activation ops in the trace IR.
    // Outside any _tracing() scope this is one TLS load + null check.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
    }

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
        if (!needs_grad)
            return out;

        // ensure_grad_fn wraps a leaf parameter in an AccumulateGrad node
        // so the autograd engine knows where to accumulate its gradient.
        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = a->shape();
        bwd->dtype_ = eff_dt;
        bwd->device_ = a->device();
        bwd->grad_dtypes_ = {detail::grad_dtype_for(a, a_ptr)};
        bwd->input_tensors_ = {a};
        // Conditionally snapshot inputs/output for use in grad_formula.
        if constexpr (Derived::kSavesInput)
            bwd->keep_input(0, a_ptr->storage());
        if constexpr (Derived::kSavesOutput) {
            bwd->keep_output(out->storage());
            bwd->bind_saved_output(out);
        }
        // Always save the original TensorImpl for graph-mode backward.
        bwd->saved_impl_inputs_ = {a};

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));
        // Version is captured so in-place modifications after this forward
        // call are detected as version mismatches during backward.
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }
}

// Out-of-class definition of :meth:`UnaryKernel::apply`.
//
// Delegates to ``Derived::grad_formula(grad_out)`` and broadcast-reduces
// the resulting :class:`Storage` back to ``input_shapes_[0]`` via
// :func:`reduce_grad_to_shape`.
template <class Derived>
std::vector<Storage> UnaryKernel<Derived>::apply(Storage grad_out) {
    if (!this->needs_input_grad(0))
        return {Storage{}};
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    Storage reduced = (this->out_shape_ == this->input_shapes_[0])
                          ? std::move(dx)
                          : reduce_grad_to_shape(dx, this->out_shape_, this->input_shapes_[0],
                                                 this->dtype_, this->device_);
    return {detail::grad_to_input_dtype(std::move(reduced), this->grad_dtypes_[0],
                                        this->input_shapes_[0], this->dtype_, this->device_)};
}

// Out-of-class definition of :meth:`UnaryKernel::apply_for_graph`.
//
// Invokes ``Derived::grad_formula_impl(grad_out, a, out)`` then reduces
// the traced gradient back to the original input shape using
// :func:`sum_op` and :func:`reshape_op` so the returned
// :class:`TensorImpl` carries a complete autograd subgraph suitable for
// higher-order differentiation.
template <class Derived>
std::vector<TensorImplPtr> UnaryKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    if (!this->saved_impl_inputs_[0]) {
        throw std::runtime_error("apply_for_graph: saved_impl_inputs_[0] not set for op '" +
                                 std::string(Derived::schema_v1.name) + "'.");
    }
    const auto a = detail::graph_input_at_kernel_dtype(this->saved_impl_inputs_[0], this->dtype_);

    // saved_impl_output_ is a WEAK ref (it breaks the node -> output -> grad_fn
    // self-cycle that would otherwise retain the whole graph in inference and
    // OOM).  For create_graph double-backward the live, grad_fn-bearing output is
    // usually still alive — pinned by a consumer's saved_impl_inputs_ or by the
    // user — and lock() returns it unchanged.
    //
    // Not every consumer pins it: ``where`` keeps only its condition.  The output
    // is then rebuilt as it was, the saved storage *with this node as its
    // grad_fn*, rather than as a data-only leaf.  A leaf made ``out`` a constant
    // in formulas that read it (``sqrt``: g / 2y), so the second derivative lost
    // dy/dx without a word — ``cdist``'s came back with the wrong sign.  The
    // rebuilt tensor holds this node only as long as the graph built here does.
    TensorImplPtr out_impl = this->saved_impl_output_.lock();
    if (!out_impl && storage_nbytes(this->saved_output_) > 0) {
        out_impl = std::make_shared<TensorImpl>(this->saved_output_, this->out_shape_, this->dtype_,
                                                this->device_, false);
        out_impl->set_grad_fn(this->shared_from_this());
        out_impl->set_grad_output_nr(0);
        out_impl->set_leaf(false);
        out_impl->set_requires_grad(true);
    }
    auto dx = static_cast<Derived*>(this)->grad_formula_impl(grad_out, a, out_impl);

    // Reduce back to input shape if needed (same as apply()).
    if (dx->shape() == this->input_shapes_[0])
        return {detail::graph_grad_to_input_dtype(dx, this->grad_dtypes_[0])};
    std::vector<int> axes;
    const int ng = static_cast<int>(dx->shape().size());
    const int nt = static_cast<int>(this->input_shapes_[0].size());
    for (int i = 0; i < ng - nt; ++i)
        axes.push_back(i);
    for (int i = 0; i < nt; ++i) {
        if (this->input_shapes_[0][static_cast<std::size_t>(i)] == 1 &&
            dx->shape()[static_cast<std::size_t>(i + ng - nt)] != 1)
            axes.push_back(i + ng - nt);
    }
    if (!axes.empty())
        dx = sum_op(dx, axes, false);
    if (dx->shape() != this->input_shapes_[0])
        dx = reshape_op(dx, this->input_shapes_[0]);
    return {detail::graph_grad_to_input_dtype(dx, this->grad_dtypes_[0])};
}

}  // namespace lucid
