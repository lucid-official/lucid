// lucid/_C/kernel/ReduceKernel.h
//
// CRTP base for single-input reduction ops (sum, mean, max, min, etc.).
// Unlike UnaryKernel, the output shape differs from the input shape so
// the backward pass must broadcast the gradient back to the full input
// extent via restore_shape() / broadcast before calling grad_formula.
//
// A concrete reduction op is declared as:
//
//   struct SumOp : ReduceKernel<SumOp> {
//       static constexpr OpSchema schema_v1 = {"sum", ...};
//       static CpuStorage cpu_kernel(const CpuStorage&, const Shape&,
//                                    const std::vector<int>& axes,
//                                    bool keepdims, Dtype);
//       static GpuStorage gpu_kernel(const GpuStorage&, const Shape&,
//                                    const std::vector<int>& axes,
//                                    bool keepdims, Dtype);
//       Storage grad_formula(Storage grad_out);
//   };
//
// The extra state members (reduce_axes_, keepdims_, full_input_shape_)
// are stored on the backward node so grad_formula can reconstruct the
// shape transformation done during forward.

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../compile/Tracer.h"  // 3.5 Phase 1.2 step 5: trace I/O + reduction attrs
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

// Concept: ``Derived`` exposes a static ``gpu_kernel`` matching the
// reduction GPU signature.
//
// Required signature
// ------------------
// ``static GpuStorage Derived::gpu_kernel(const GpuStorage& a,
// const Shape& in_shape, const std::vector<int>& axes, bool keepdims,
// Dtype dt)``.
//
// Notes
// -----
// When satisfied, ``ReduceKernel::forward`` dispatches GPU storage
// directly through this static instead of routing through
// :class:`backend::IBackend`.  Concepts are checked at the call site
// with ``if constexpr`` so non-conforming derived classes silently fall
// through to the dispatch-interface or CPU branches.
template <class T>
concept HasReduceGpuKernel =
    requires(GpuStorage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::gpu_kernel(a, s, ax, kd, d) } -> std::same_as<GpuStorage>;
    };

// Concept: ``Derived`` routes through :class:`backend::IBackend` for
// the forward reduction.
//
// Required signature
// ------------------
// ``static Storage Derived::dispatch(backend::IBackend& be,
// const Storage& a, const Shape& in_shape, const std::vector<int>& axes,
// bool keepdims, Dtype dt)``.
//
// Notes
// -----
// This is the preferred wiring style for canonical reductions
// (``SumBackward``, ``MeanBackward``, …) because it lets a single
// derived class drive both CPU (Accelerate) and GPU (MLX) by delegating
// to the per-device backend implementation.  Takes precedence over
// :concept:`HasReduceGpuKernel` when both are satisfied.
template <class T>
concept HasReduceDispatch =
    requires(backend::IBackend& be, Storage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::dispatch(be, a, s, ax, kd, d) } -> std::same_as<Storage>;
    };

}  // namespace detail

// CRTP base for single-input multi-axis reduction ops (``sum``,
// ``mean``, ``prod``, ``max``, ``min``, …).
//
// Concrete reductions inherit as
// ``class SumBackward : public ReduceKernel<SumBackward>`` and supply
// (a) a ``static constexpr OpSchema schema_v1``, (b) one of the kernel
// hooks recognised by :concept:`HasReduceDispatch` /
// :concept:`HasReduceGpuKernel` (or a plain ``cpu_kernel`` static), and
// (c) a ``grad_formula(grad_out) -> Storage`` that converts the upstream
// gradient into the input gradient.  The base owns axis normalisation,
// output-shape computation, GPU/CPU dispatch, and the extra
// reduction-state plumbing that the backward formula needs.
//
// Unlike :class:`UnaryKernel`, the forward output has a *different*
// shape than the input, so the backward pass must broadcast the
// gradient from the reduced shape back to the full input extent.  That
// broadcast is encoded once in :meth:`apply_for_graph` (graph-mode) and
// is the responsibility of ``grad_formula`` in storage-mode.
//
// Math
// ----
// $$
//   y = \mathrm{reduce}_{i \in \text{axes}}(x), \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i}
//     = \mathrm{scale}_{\text{op}}\!\left(
//       \mathrm{broadcast}\!\left(
//         \frac{\partial \mathcal{L}}{\partial y}
//       \right)
//     \right).
// $$
//
// ``scale_op`` is identity for ``sum``, $1/N$ for ``mean``,
// $y / x_i$ for ``prod``, an argmax/argmin indicator for ``max``/``min``.
//
// Template Parameters
// -------------------
// Derived : class
//     CRTP self-type.  Must expose ``schema_v1``, a kernel hook
//     satisfying one of the reduction concepts (or ``cpu_kernel``), and
//     ``grad_formula``.  May override ``scale_graph_grad`` for graph-mode
//     scaling (``mean``, ``prod``, …).
//
// Attributes
// ----------
// kSavesInput : bool
//     ``true`` by default — most reductions need the input activation
//     during backward.  ``SumBackward`` overrides this to ``false``.
// kSavesOutput : bool
//     ``false`` by default.  ``MaxBackward`` / ``MinBackward`` override
//     to ``true`` so they can route gradient through the saved output's
//     argmax indicator.
// kHasGradient : bool
//     ``true``.  All canonical reductions participate in autograd.
// reduce_axes_ : std::vector<int>
//     Normalised (non-negative, sorted, deduplicated) axis list saved by
//     :meth:`forward` for use in :meth:`apply_for_graph` and in the
//     derived ``grad_formula``.
// keepdims_ : bool
//     Whether reduced axes were retained as size-1 dims in the output.
//     Controls whether the backward pass must re-insert those size-1
//     axes before broadcasting.
// full_input_shape_ : Shape
//     Shape of the input tensor *before* reduction — the broadcast
//     target during backward.
//
// Notes
// -----
// **Dispatch order.**  :meth:`forward` chooses the kernel path with
// ``if constexpr``:
//
// 1. :concept:`HasReduceDispatch` — call ``Derived::dispatch(backend,
//    storage, shape, axes, keepdims, dtype)``.
// 2. GPU + :concept:`HasReduceGpuKernel` — call ``Derived::gpu_kernel``.
// 3. CPU — call ``Derived::cpu_kernel``.
//
// **Contiguity.**  A strided CPU input needs no copy first: its
// ``storage()`` is packed.  GPU inputs are passed through as MLX handles
// strides natively.
//
// See Also
// --------
// :class:`UnaryKernel`, :class:`BinaryKernel`, :class:`NaryKernel` —
// sibling CRTP bases for shape-preserving ops.
// :class:`SumBackward`, :class:`MeanBackward`, :class:`ProdBackward`,
// :class:`MaxBackward`, :class:`MinBackward` — canonical consumers.
template <class Derived>
class ReduceKernel : public AutogradNode<Derived, 1>, public kernel::IKernel {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    // Schema-derived op name, exposed to the :class:`IKernel` virtual
    // interface for error messages and profiler scopes.
    //
    // Returns
    // -------
    // std::string_view
    //     ``Derived::schema_v1.name`` (e.g. ``"sum"``, ``"mean"``).
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Schema-derived node name used by the autograd graph display and
    // ``create_graph`` error reports.
    //
    // Returns
    // -------
    // std::string
    //     Owning copy of ``Derived::schema_v1.name``.
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Normalised reduction axes captured during :meth:`forward`.
    //
    // Sorted, non-negative, and deduplicated.  ``grad_formula`` and
    // :meth:`apply_for_graph` use this to invert the shape change.
    std::vector<int> reduce_axes_;
    // Whether ``keepdims=True`` was requested for the forward call.
    //
    // When ``false``, the backward pass must re-insert size-1 axes at
    // the reduced positions before broadcasting back to
    // :attr:`full_input_shape_`.
    bool keepdims_ = false;
    // Original input shape — the broadcast target for the gradient.
    //
    // Captured before any contiguity / dtype-cast hops so the broadcast
    // matches the user's view of the tensor, not the kernel's internal
    // representation.
    Shape full_input_shape_;

    // Graph-mode gradient scaling hook.
    //
    // Default implementation passes the gradient through unchanged,
    // matching the ``sum`` rule (every reduced element contributed with
    // unit weight).  Derived classes whose rule has a non-trivial scale
    // — :class:`MeanBackward` divides by $N$, :class:`ProdBackward`
    // multiplies by the product of the others, the argmax variants apply an indicator
    // mask — override this method to inject their factor before the
    // gradient is returned.
    //
    // Parameters
    // ----------
    // g : const TensorImplPtr&
    //     Broadcast-back gradient already shaped like the original
    //     input.  Always non-null.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     The scaled gradient — either ``g`` itself or a fresh tensor
    //     produced by a graph-tracked op.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g) { return g; }

    // Forward entry point: run the reduction and wire the backward node.
    //
    // Normalises ``axes_user`` (negative indices → positive, dedup),
    // computes the output shape (respecting ``keepdims``), enforces CPU
    // contiguity, dispatches to the most specific kernel hook the
    // ``Derived`` class exposes, and — when ``GradMode`` is enabled and
    // the input requires grad — constructs the backward node with the
    // extra reduction state (:attr:`reduce_axes_`, :attr:`keepdims_`,
    // :attr:`full_input_shape_`) populated.
    //
    // Parameters
    // ----------
    // a : const std::shared_ptr<TensorImpl>&
    //     Input tensor.  Must be non-null.
    // axes_user : const std::vector<int>&
    //     Raw axis list as supplied by the caller.  May contain
    //     negative indices and duplicates; they are normalised before
    //     dispatch.  An empty list reduces over every axis.
    // keepdims : bool
    //     If ``true``, every reduced axis appears as a size-1 dim in
    //     the output; otherwise it is removed.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     Output tensor with the reduced shape and the AMP-effective
    //     dtype (driven by ``schema_v1.amp_policy``).
    //
    // Raises
    // ------
    // LucidError
    //     If ``a`` is null.
    // NotImplemented
    //     If the input lives on GPU and ``Derived`` exposes neither
    //     :concept:`HasReduceDispatch` nor :concept:`HasReduceGpuKernel`.
    static std::shared_ptr<TensorImpl>
    forward(const std::shared_ptr<TensorImpl>& a, const std::vector<int>& axes_user, bool keepdims);

    // Storage-mode backward: delegate the full broadcast-and-scale to
    // ``Derived::grad_formula``.
    //
    // Unlike :class:`UnaryKernel::apply`, this method performs no
    // ``reduce_grad_to_shape`` step — the reduction kernel's
    // ``grad_formula`` is responsible for broadcasting the upstream
    // gradient back to :attr:`full_input_shape_` using the saved
    // :attr:`reduce_axes_` and :attr:`keepdims_` fields, and for any
    // reduction-specific scaling.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Gradient of the loss with respect to this op's output, shaped
    //     like the forward output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element vector containing the input gradient, shaped
    //     like :attr:`full_input_shape_`.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: rebuild the broadcast in tracked ops.
    //
    // Re-inserts size-1 axes for any dims that were dropped because
    // ``keepdims=False``, runs :func:`broadcast_to_op` to expand the
    // gradient back to :attr:`full_input_shape_`, then defers to
    // :meth:`scale_graph_grad` for the reduction-specific factor.  The
    // entire chain participates in the autograd graph so higher-order
    // gradients work correctly.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient as a live :class:`TensorImpl` (i.e. a graph
    //     node), not just a :class:`Storage`.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     Single-element vector containing the input gradient tensor.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

protected:
    // The reduced storage of ``a_ptr`` over ``axes`` at ``eff_dt``: the
    // backend's dispatch, or the Derived CPU / GPU kernel.  A phantom of
    // ``out_shape`` under phantom mode, where no kernel runs.
    static Storage compute_storage(const TensorImplPtr& a_ptr,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   const Shape& out_shape,
                                   Dtype eff_dt);
};

template <class Derived>
Storage ReduceKernel<Derived>::compute_storage(const TensorImplPtr& a_ptr,
                                               const std::vector<int>& axes,
                                               bool keepdims,
                                               const Shape& out_shape,
                                               Dtype eff_dt) {
    if (PhantomMode::is_enabled()) [[unlikely]]
        return phantom_storage(out_shape, eff_dt, a_ptr->device());
    if constexpr (detail::HasReduceDispatch<Derived>) {
        return Derived::dispatch(backend::Dispatcher::for_device(a_ptr->device()), a_ptr->storage(),
                                 a_ptr->shape(), axes, keepdims, eff_dt);
    } else {
        if (a_ptr->device() == Device::GPU) {
            if constexpr (detail::HasReduceGpuKernel<Derived>) {
                return Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()),
                                                   a_ptr->shape(), axes, keepdims, eff_dt)};
            } else {
                ErrorBuilder(Derived::schema_v1.name)
                    .not_implemented("GPU kernel not yet implemented");
            }
        }
        return Storage{Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()), a_ptr->shape(),
                                           axes, keepdims, eff_dt)};
    }
}

// Out-of-line definition of :meth:`ReduceKernel::forward`.
//
// Pipeline
// --------
// 1. Validate ``a`` is non-null.
// 2. Enter the :class:`SchemaGuard` to resolve the AMP-effective dtype.
// 3. On CPU, read the input through ``storage()``, which packs a strided
//    view; on GPU, pass through unchanged.
// 4. Cast inputs to ``eff_dt`` via ``maybe_cast_for_kernel`` when AMP
//    requires it.
// 5. Normalise the user axis list and compute the reduced output shape
//    (``keepdims`` controls whether reduced axes become size-1 or are
//    dropped entirely).
// 6. Dispatch in priority order: ``Derived::dispatch`` →
//    ``Derived::gpu_kernel`` (GPU only) → ``Derived::cpu_kernel``.
// 7. If gradient tracking is active, allocate the backward node, copy
//    saved inputs / output per the ``kSavesInput`` / ``kSavesOutput``
//    flags, store the extra reduction metadata
//    (:attr:`reduce_axes_`, :attr:`keepdims_`, :attr:`full_input_shape_`),
//    wire next-edges, and attach the node to the output.
//
// See class-level docstring for parameter / return contract.
template <class Derived>
std::shared_ptr<TensorImpl> ReduceKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                           const std::vector<int>& axes_user,
                                                           bool keepdims) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    // A strided CPU input needs no copy here — see UnaryKernel.
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a, eff_dt);

    // normalize_axes wraps negative indices and refuses a repeated axis;
    // the result is a sorted, non-negative axis list for the kernels.
    const std::string op_name(Derived::schema_v1.name);
    const auto axes =
        normalize_axes(axes_user, static_cast<int>(a_ptr->shape().size()), op_name.c_str());
    Shape out_shape = reduce_output_shape(a_ptr->shape(), axes, keepdims);

    // max and min have no identity element, so an empty reduced axis has no
    // answer: refused as the reference refuses it.  The CPU answered -inf
    // (or +inf, or an int's limit) as if it were a value, and Metal raised
    // MLX's own message.  A *non*-empty reduced axis of an empty tensor is
    // fine — the result is empty too.
    if constexpr (requires { Derived::kNoIdentity; }) {
        for (int ax : axes) {
            if (a_ptr->shape()[static_cast<std::size_t>(ax)] != 0)
                continue;
            if (axes_user.empty())
                ErrorBuilder(Derived::schema_v1.name)
                    .fail("Expected reduction dim to be specified for input.numel() == 0. "
                          "Specify the reduction dim with the 'dim' argument.");
            ErrorBuilder(Derived::schema_v1.name)
                .index_error("Expected reduction dim " + std::to_string(ax) +
                             " to have non-zero size.");
        }
    }

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, out_shape};

    Storage out_storage = compute_storage(a_ptr, axes, keepdims, out_shape, eff_dt);

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, eff_dt, a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    // 3.5 Phase 1.2 step 5: trace I/O wiring at the forward boundary
    // (ReduceKernel doesn't go through NaryKernel::wire_autograd) +
    // report the reduction axes / keepdim flag so the Reduction
    // emitter can rebuild the same reduction inside MPSGraph.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
        std::vector<std::int64_t> dims64(axes.begin(), axes.end());
        trc->on_op_attr("dims", std::move(dims64));
        trc->on_op_attr("keepdim", static_cast<bool>(keepdims));
    }

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
        if (!needs_grad)
            return out;

        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = out_shape;
        bwd->dtype_ = eff_dt;
        bwd->device_ = a->device();
        bwd->grad_dtypes_ = {detail::grad_dtype_for(a, a_ptr)};
        bwd->input_tensors_ = {a};
        if constexpr (Derived::kSavesInput)
            bwd->keep_input(0, a_ptr->storage());
        if constexpr (Derived::kSavesOutput) {
            bwd->keep_output(out->storage());
            bwd->bind_saved_output(out);
        }
        bwd->saved_impl_inputs_ = {a};

        // Store reduction metadata so grad_formula can invert the shape change.
        bwd->reduce_axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->full_input_shape_ = a->shape();

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }
}

// Out-of-line definition of :meth:`ReduceKernel::apply`.
//
// Delegates the entire backward computation to
// ``Derived::grad_formula(grad_out)``.  No ``reduce_grad_to_shape``
// step is performed here (in contrast to :meth:`UnaryKernel::apply`),
// because the reduction's gradient already needs a broadcast — the
// opposite shape transformation — and that broadcast logic lives inside
// each derived ``grad_formula``.
//
// See class-level docstring for parameter / return contract.
template <class Derived>
std::vector<Storage> ReduceKernel<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {detail::grad_to_input_dtype(std::move(dx), this->grad_dtypes_[0],
                                        this->full_input_shape_, this->dtype_, this->device_)};
}

// Out-of-line definition of :meth:`ReduceKernel::apply_for_graph`.
//
// Pipeline
// --------
// 1. If ``keepdims_ == false``, re-insert size-1 axes at each reduced
//    position by chaining :func:`unsqueeze_op` over a sorted axis list.
//    Sorting ensures earlier inserts do not shift the indices of later
//    ones.
// 2. Call :func:`broadcast_to_op` to expand the gradient up to
//    :attr:`full_input_shape_` — for ``sum`` this is the full backward
//    rule, since every reduced element contributed with unit weight.
// 3. Defer to :meth:`scale_graph_grad` for the reduction-specific
//    scale factor (identity for ``sum``, $1/N$ for ``mean``, …).
//
// All three steps build tracked graph nodes so second-order derivatives
// flow naturally through the reduction.
//
// See class-level docstring for parameter / return contract.
template <class Derived>
std::vector<TensorImplPtr> ReduceKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    extern TensorImplPtr broadcast_to_op(const TensorImplPtr&, const Shape&);
    extern TensorImplPtr unsqueeze_op(const TensorImplPtr&, int);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    TensorImplPtr g = grad_out;

    // If keepdims was false, the reduced axes are missing from grad_out.
    // Re-insert size-1 axes at the correct positions before broadcasting.
    if (!this->keepdims_) {
        std::vector<int> sorted_axes = this->reduce_axes_;
        std::sort(sorted_axes.begin(), sorted_axes.end());
        for (int ax : sorted_axes) {
            g = unsqueeze_op(g, ax);
        }
    }

    // Broadcast to the original input shape — this replicates the gradient
    // to all elements that were reduced (sum backward).
    auto dx = broadcast_to_op(g, this->full_input_shape_);

    // Derived may scale the gradient (e.g. MeanBackward divides by n_reduced)
    // and reads the saved input there, which must be at the kernel dtype like
    // ``dx``.  The cast stands in for the call only: left in the slot, a
    // retained graph's next restore_saved_for_graph would take it for an
    // input moved by an in-place write.
    struct RestoreSaved {
        TensorImplPtr& slot;
        TensorImplPtr saved;
        ~RestoreSaved() { slot = std::move(saved); }
    } restore{this->saved_impl_inputs_[0], this->saved_impl_inputs_[0]};
    this->saved_impl_inputs_[0] = detail::graph_input_at_kernel_dtype(restore.saved, this->dtype_);
    return {detail::graph_grad_to_input_dtype(static_cast<Derived*>(this)->scale_graph_grad(dx),
                                              this->grad_dtypes_[0])};
}

}  // namespace lucid
