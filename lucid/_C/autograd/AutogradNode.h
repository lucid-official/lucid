// lucid/_C/autograd/AutogradNode.h
//
// CRTP base class for every concrete backward node in the engine's ``ops/``
// layer.
//
// :class:`AutogradNode` extends :class:`Node` with the boilerplate that
// almost every op shares: fixed-size arrays of saved input / output
// :class:`Storage`, weak pointers to the source :class:`TensorImpl` objects
// for version checking, and the shape / dtype / device metadata needed to
// rebuild gradient tensors during backward.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

class TensorImpl;

// ``reads[i][j]``: whether the gradient of input ``i`` reads input ``j``'s
// values.  A node declares it as ``static constexpr GradReads<N> kGradReads``
// when its gradients do not each read every input it saved — ``mul`` reads
// the other operand only — so a write into an input that only an unneeded
// gradient reads is not refused (see :meth:`AutogradNode::validate_versions`).
template <std::size_t N>
using GradReads = std::array<std::array<bool, N>, N>;

// CRTP base supplying boilerplate fields and methods to concrete backward
// nodes with a fixed number of forward inputs.
//
// Inherit as ``class FooBackward : public AutogradNode<FooBackward, N>`` —
// or, more commonly, through the :class:`FuncOp` alias — to obtain
// ``N``-slot saved-input storage, automatic version validation, and an
// engine-friendly ``release_saved`` implementation.  The derived class
// must expose a ``static constexpr OpSchema schema_v1`` whose ``.name``
// field identifies the op in error messages and traces, and must
// implement :meth:`apply` returning one :class:`Storage` per forward
// input.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete backward node class (CRTP self-type).  Must expose a
//     ``static constexpr`` member ``schema_v1`` with at least a ``.name``
//     :class:`std::string_view` field.
// N_IN : std::size_t
//     Number of input tensors the corresponding forward op consumes.
//     Determines the compile-time size of :attr:`saved_inputs_`,
//     :attr:`input_shapes_`, :attr:`input_tensors_`, and
//     :attr:`saved_impl_inputs_`.
//
// Attributes
// ----------
// kNumInputs : std::size_t
//     Compile-time constant mirror of ``N_IN`` for callers that need it
//     as a value (e.g. when constructing the ``next_edges`` vector).
// input_tensors_ : std::array<std::weak_ptr<TensorImpl>, N_IN>
//     Weak references to the original input :class:`TensorImpl` objects,
//     used by :meth:`validate_versions` for the inputs saved by value.
//     Weak so the backward graph never extends an input's lifetime.
// input_shapes_ : std::array<Shape, N_IN>
//     Shapes of the forward inputs, captured so that broadcasts and
//     reductions can be inverted in :meth:`apply` without needing the
//     live :class:`TensorImpl`.
// out_shape_ : Shape
//     Shape of the forward output; used to allocate or validate the
//     incoming gradient in :meth:`apply`.
// dtype_ : Dtype
//     Dtype of the forward computation.  Gradient tensors are created
//     with this dtype.  Defaults to :data:`Dtype::F32`.
// device_ : Device
//     Device of the forward computation.  Gradient tensors live on the
//     same device.  Defaults to :data:`Device::CPU`.
// grad_dtypes_ : std::array<std::optional<Dtype>, N_IN>
//     The dtype input ``i`` had before the forward cast it to
//     :attr:`dtype_` for an AMP policy; empty when it was used as given.
//     The cast is part of the op, so the op hands that input's gradient
//     back in this dtype — a view or permute between the input and this
//     node reads it at its own width.  Set only for inputs that take a
//     gradient.
// saved_inputs_ : std::array<Storage, N_IN>
//     Copies of forward input :class:`Storage` values that :meth:`apply`
//     reads — for example, the input activations of an element-wise
//     multiply.  Populated by the forward op builder.  The one place a
//     node keeps an input's values: a slot holding them is what
//     :meth:`validate_versions` guards and :meth:`restore_saved_for_graph`
//     rebuilds from, so a node never keeps an input's values in a member of
//     its own (``tools/check_saved_versions.py``).
// saved_output_ : Storage
//     Copy of the forward output :class:`Storage`, populated only when
//     the backward formula needs the output activation (sigmoid,
//     softmax, tanh, etc.).  Left as an empty :class:`CpuStorage` when
//     unused.
// saved_impl_inputs_ : std::array<std::shared_ptr<TensorImpl>, N_IN>
//     Strong references to the source :class:`TensorImpl` objects.
//     Set in lockstep with :attr:`saved_inputs_` so that
//     :meth:`Node::apply_for_graph` can call into ``grad_fn``-bearing
//     ops when ``create_graph=True``.
// saved_impl_output_ : std::weak_ptr<TensorImpl>
//     **Weak** reference to the forward output :class:`TensorImpl`, set
//     only for ops whose graph-mode backward needs the output value
//     (e.g. :class:`SigmoidBackward`).  Deliberately weak: a strong ref
//     here would form a ``Node -> output -> grad_fn -> same Node`` cycle
//     (the output already owns this node via ``grad_fn``), and since the
//     cycle is broken only by ``backward()`` clearing ``grad_fn``, an
//     inference / forward-only loop that never calls backward would
//     retain every per-call graph and eventually OOM.  Holding it weakly
//     keeps the autograd reference graph a pure backward-pointing DAG, so
//     dropping the output cascades the whole graph free.  ``create_graph``
//     backward does not read the live output through it any more: a write
//     after forward could have changed it with nothing to show, so
//     :meth:`restore_saved_for_graph` drops the handle and the reader
//     rebuilds the output from :attr:`saved_output_` with this node as its
//     grad_fn — the case it already covered for an output that was gone.
//
// Notes
// -----
// **Lifecycle.** The forward op builder constructs an instance, fills
// :attr:`input_tensors_`, :attr:`input_shapes_`, :attr:`saved_inputs_`,
// :attr:`saved_output_`, :attr:`dtype_`, and :attr:`device_`, then calls
// :meth:`Node::set_next_edges` and :meth:`Node::set_saved_versions`
// before attaching the node to the output :class:`TensorImpl` via
// ``set_grad_fn``.  During backward the engine calls
// :meth:`validate_versions` then :meth:`apply`.  After :meth:`apply`
// returns, the engine calls :meth:`release_saved`, which resets every
// :class:`Storage` slot to a default-constructed (zero-byte)
// :class:`CpuStorage` so referenced memory can be freed immediately.
//
// **Thread safety.** None.  All access happens on the single backward
// thread.
//
// **Slot ordering.** :attr:`saved_inputs_` and the gradients returned
// from :meth:`apply` are ordered to match the forward op's positional
// argument list — changing that order is a wire-format break.
//
// See Also
// --------
// :class:`Node` — the abstract base whose virtuals this class overrides.
// :class:`FuncOp` — alias used by most concrete backward classes.
template <typename... S>
inline void clear_saved(S&... s) {
    ((s = S{}), ...);
}

template <class Derived, std::size_t N_IN>
class AutogradNode : public Node {
public:
    // Compile-time count of forward inputs.  Equivalent to ``N_IN``;
    // exposed as a value so non-template callers can read it.
    static constexpr std::size_t kNumInputs = N_IN;

    // Human-readable name of the op, sourced from ``Derived::schema_v1``.
    //
    // Returns
    // -------
    // std::string_view
    //     The schema name (e.g. ``"linear"``, ``"matmul"``).  Used in
    //     error messages, profiler traces, and ``create_graph`` failure
    //     reports.
    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    // Verify that no input whose values :meth:`apply` reads has been
    // written in place since forward.
    //
    // An input is guarded when its values sit in :attr:`saved_inputs_`
    // (:func:`holds_saved_values`) — the slot a backward that reads them
    // saves them in — and a gradient that is needed reads them: one whose
    // input takes a gradient (a live edge), by :func:`grad_reads`.  The
    // others are not: ``add``, ``sum`` or a view read at most their shape,
    // and ``x * c`` with a constant ``c`` reads ``x`` only for ``c``'s
    // gradient, which nothing asks for.  Refusing a write into one of them
    // refused what the reference allows.  Expired weak references are
    // skipped — if the input has been destroyed, no consumer can observe a
    // corrupted gradient.
    //
    // Raises
    // ------
    // VersionMismatch
    //     When a guarded input's live version counter differs from the
    //     value captured at forward.  The error names this op (via
    //     ``Derived::schema_v1.name``) and the offending input index.
    void validate_versions() override {
        for (std::size_t j = 0; j < N_IN; ++j) {
            if (::lucid::holds_saved_values(saved_inputs_[j]) && read_by_a_needed_gradient(j))
                check_saved_version(input_tensors_[j], j, Derived::schema_v1.name,
                                    argument_position(j));
        }
        if (saved_output_version_sink_ && any_needed_gradient()) {
            check_version_match(saved_output_version_sink_, saved_output_version_,
                                std::string(Derived::schema_v1.name) + " output 0");
        }
    }

    // The forward argument slot ``j`` stands for, which the error names:
    // ``Derived::kArgPositions`` where the node is wired on fewer inputs
    // than its op takes (``lu_solve(LU, pivots, B)`` wires ``{LU, B}``),
    // otherwise ``j`` itself.
    static constexpr std::size_t argument_position(std::size_t j) {
        if constexpr (requires { Derived::kArgPositions; })
            return Derived::kArgPositions[j];
        else
            return j;
    }

    // Whether the gradient of input ``i`` reads input ``j``'s values:
    // ``Derived::kGradReads`` where the node declares it, otherwise every
    // gradient is taken to read every input the node saved.
    static constexpr bool grad_reads(std::size_t i, std::size_t j) {
        if constexpr (requires { Derived::kGradReads; })
            return Derived::kGradReads[i][j];
        else
            return true;
    }

    // Whether input ``i`` has a valid outgoing autograd edge (requires grad).
    bool needs_input_grad(std::size_t i) const noexcept {
        const auto& edges = next_edges();
        return i < edges.size() && edges[i].is_valid();
    }

    // Put back, for graph-mode backward, every saved tensor a write since
    // forward may have changed without :meth:`validate_versions` refusing.
    //
    // An input is replaced by the tensor it was at forward time, rebuilt
    // from :attr:`saved_inputs_` and its edge (:func:`forward_time_input`),
    // when it no longer sits where it did — reachable past the version check
    // only while that check is waived.  The output is always rebuilt — from
    // :attr:`saved_output_`, with this node as its grad_fn — by dropping the
    // weak handle, which is the case a reader already covers for an output
    // that is gone: no version guards it, and a write under ``no_grad``
    // leaves its grad_fn where it was, so nothing on the live tensor says
    // whether it still holds the values this node saw.  Either way
    // graph-mode backward reads the values eager backward reads.
    //
    // Raises
    // ------
    // LucidError
    //     An input was written into and its saved values cannot rebuild it.
    void restore_saved_for_graph() override {
        const auto& edges = next_edges();
        for (std::size_t i = 0; i < N_IN && i < edges.size(); ++i) {
            if (auto then = ::lucid::forward_time_input(saved_impl_inputs_[i], saved_inputs_[i],
                                                        input_shapes_[i], edges[i],
                                                        Derived::schema_v1.name, i))
                saved_impl_inputs_[i] = std::move(then);
        }
        if (storage_nbytes(saved_output_) > 0)
            saved_impl_output_.reset();
    }

    // Drop every reference held for backward so that memory can be
    // reclaimed immediately after the engine finishes with this node.
    //
    // Resets all entries of :attr:`saved_inputs_`, clears
    // :attr:`saved_output_`, empties the :attr:`input_tensors_` weak
    // array, releases the strong :class:`TensorImpl` references in
    // :attr:`saved_impl_inputs_`, and clears the weak
    // :attr:`saved_impl_output_` handle.
    //
    // Notes
    // -----
    // Called by :class:`Engine` once :meth:`apply` returns, except when
    // ``retain_graph=True``.  After this call the node retains only its
    // sequence number and (now-empty) metadata; calling :meth:`apply`
    // again is undefined.
    void release_saved() override {
        for (auto& s : saved_inputs_)
            s = Storage{CpuStorage{}};
        input_saved_.fill(false);
        saved_output_ = Storage{CpuStorage{}};
        saved_output_version_sink_.reset();
        keeps_output_ = false;
        input_tensors_ = {};
        for (auto& p : saved_impl_inputs_)
            p.reset();
        saved_impl_output_.reset();
        if constexpr (requires(Derived& d) { d.release_own_saved(); }) {
            static_cast<Derived*>(this)->release_own_saved();
        }
    }

    // Keep ``s`` as input ``i``'s values, or the output's, for :meth:`apply`.
    // The slot is recorded as kept whatever its size: an eager pass reads it
    // even for a tensor of no elements — on Metal as the GPU buffer a
    // release replaces — so a released one must be refused, as the
    // reference refuses it.
    void keep_input(std::size_t i, Storage s) {
        saved_inputs_[i] = std::move(s);
        input_saved_[i] = true;
        keeps_values_ = true;
    }
    void keep_output(Storage s) {
        saved_output_ = std::move(s);
        keeps_values_ = true;
        keeps_output_ = true;
    }
    void bind_saved_output(const TensorImplPtr& out) {
        if (!keeps_output_ || !out)
            return;
        saved_output_ = out->saved_output_storage();
        saved_impl_output_ = out;
        saved_output_version_ = out->version();
        saved_output_version_sink_ = out->version_sink();
    }
    // Every input's values at once.
    void keep_inputs(std::array<Storage, N_IN> s) {
        saved_inputs_ = std::move(s);
        input_saved_.fill(true);
        keeps_values_ = true;
    }

    bool any_needed_gradient() const noexcept {
        for (const auto& edge : next_edges()) {
            if (edge.is_valid())
                return true;
        }
        return false;
    }

    // Whether some input that takes a gradient has one reading input ``j``.
    bool read_by_a_needed_gradient(std::size_t j) const {
        const auto& edges = next_edges();
        for (std::size_t i = 0; i < N_IN && i < edges.size(); ++i) {
            if (edges[i].is_valid() && grad_reads(i, j))
                return true;
        }
        return false;
    }

    // Keep input ``i`` for backward when ``saves`` is false (recording only
    // ``impl`` for graph-mode metadata) or when some needed gradient reads
    // input ``i`` (saving both ``storage_src->storage()`` and ``impl``).
    void keep_needed_input(std::size_t i,
                           const TensorImplPtr& impl,
                           const TensorImplPtr& storage_src,
                           bool saves) {
        if (!impl)
            return;
        if (!saves) {
            saved_impl_inputs_[i] = impl;
        } else if (read_by_a_needed_gradient(i)) {
            keep_input(i, storage_src->storage());
            saved_impl_inputs_[i] = impl;
        }
    }

    // Return ``saved_inputs_[i]`` when it was kept, or a zero-filled storage
    // of ``input_shapes_[i]`` when no needed gradient reads input ``i`` (used
    // by fused multi-output backward kernels whose unneeded output slots are
    // discarded).
    Storage saved_or_zeros(std::size_t i, const Storage& grad_out) const {
        if (input_saved_[i])
            return saved_inputs_[i];
        if (storage_is_phantom(grad_out))
            return phantom_storage(input_shapes_[i], dtype_, device_);
        return make_zero_storage(input_shapes_[i], dtype_, device_);
    }

    // :meth:`apply` reads the values this node kept: those in its slots
    // (:meth:`keep_input`, :meth:`keep_output`), and those a node keeps in
    // members of its own, which it declares with ``kKeepsOwnValues``
    // (``tools/check_saved_versions.py`` requires it of every node class
    // holding a tensor of its own).  A node that kept none frees just the
    // tensors graph-mode backward reads, and an eager pass may run it again.
    SavedKind saved_kind() const noexcept override {
        static_assert(
            std::is_same_v<decltype(&Derived::release_saved), void (AutogradNode::*)()> ||
                !std::is_same_v<decltype(&Derived::saved_kind),
                                SavedKind (AutogradNode::*)() const noexcept>,
            "a node that frees members of its own in release_saved must say in saved_kind "
            "whether its eager backward reads them");
        if constexpr (requires { Derived::kKeepsOwnValues; }) {
            static_assert(
                !Derived::kKeepsOwnValues || requires(Derived& d) { d.release_own_saved(); },
                "a node with kKeepsOwnValues = true must free its own members in "
                "release_own_saved()");
            if (Derived::kKeepsOwnValues)
                return SavedKind::kEagerValues;
        }
        bool holds = keeps_values_ || ::lucid::holds_saved_values(saved_output_);
        for (const auto& s : saved_inputs_)
            holds = holds || ::lucid::holds_saved_values(s);
        return holds ? SavedKind::kEagerValues : SavedKind::kGraphInputs;
    }

    // The inputs kept for graph-mode backward; the output is held weakly.
    void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const override {
        for (const auto& p : saved_impl_inputs_) {
            if (p)
                fn(p);
        }
    }

    void advance_saved_output_version(std::int64_t v) noexcept override {
        if (saved_output_version_sink_) {
            saved_output_version_sink_->store(static_cast<std::uint64_t>(v),
                                              std::memory_order_relaxed);
            saved_output_version_ = v;
        }
    }

    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;
    std::array<Shape, N_IN> input_shapes_;
    Shape out_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;
    std::array<std::optional<Dtype>, N_IN> grad_dtypes_{};
    std::array<Storage, N_IN> saved_inputs_;
    Storage saved_output_;
    std::array<std::shared_ptr<TensorImpl>, N_IN> saved_impl_inputs_;
    std::weak_ptr<TensorImpl> saved_impl_output_;
    std::shared_ptr<VersionCounter> saved_output_version_sink_;
    std::int64_t saved_output_version_ = 0;

private:
    // Whether :meth:`keep_input` or :meth:`keep_output` filled a slot.
    bool keeps_values_ = false;
    bool keeps_output_ = false;
    std::array<bool, N_IN> input_saved_{};
};

}  // namespace lucid
