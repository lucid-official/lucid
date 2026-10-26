// lucid/_C/autograd/CustomFunction.h
//
// Bridges the C++ autograd engine to user-defined Python backward functions
// written against the lucid.autograd.Function API.  Provides two classes and
// one registration helper:
//
//   FunctionCtx  — the context object passed to both the user's forward and
//                  backward methods; stores saved tensors and arbitrary Python
//                  key/value pairs.
//
//   PythonBackwardNode — a Node subclass that wraps a Python callable and
//                  invokes it (with the GIL held) when the Engine calls
//                  apply() during a backward pass.
//
//   register_custom_function() — registers both types and the
//                  _register_python_backward_node helper into a pybind11
//                  module so that Python code can wire custom ops into the
//                  autograd graph.

#pragma once

#include <pybind11/pybind11.h>

#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "GraphBarrier.h"
#include "Node.h"

namespace py = pybind11;

namespace lucid {

// Per-call context shared between a custom op's forward and backward passes.
//
// ``FunctionCtx`` is the C++-side counterpart of :class:`lucid.autograd.Function`'s
// Python ``ctx`` object.  A fresh instance is created on each ``Function.apply``
// invocation and is passed unchanged as the first argument to both
// ``forward(ctx, ...)`` and ``backward(ctx, grad_output)``, providing the only
// legal channel for state to cross the forward/backward boundary.
//
// Attributes
// ----------
// saved_tensors_ : std::vector<std::shared_ptr<TensorImpl>>
//     Tensors stored via :func:`save_for_backward` as strong references so
//     they survive until the backward pass executes.
// extras_ : std::unordered_map<std::string, py::object>
//     Generic Python key/value bag mirroring ``ctx.<name> = value`` /
//     ``ctx.<name>`` access from Python.  Used for scalars, shape tuples,
//     hyperparameters, and other non-tensor state.
//
// Notes
// -----
// The object is heap-allocated and jointly owned by the Python wrapper and
// the associated :class:`PythonBackwardNode`, so it outlives the forward
// call and remains valid for the entire backward traversal.  Capturing
// tensors through Python closures instead of saving them through ``ctx``
// bypasses this ownership tracking and leaks memory.
//
// See Also
// --------
// :class:`lucid.autograd.FunctionCtx` — Python-facing wrapper class.
// :class:`PythonBackwardNode` — owns one ``FunctionCtx`` per custom op call.
class LUCID_API FunctionCtx {
public:
    // Construct a ``FunctionCtx`` for a custom ``Function::forward`` call.
    //
    // The newly created context starts with empty ``saved_tensors_`` and
    // ``extras_`` containers; the user's ``forward`` body populates them
    // via :func:`save_for_backward` and :func:`store` before returning.
    // Defaulted — no resources need acquisition at construction.
    FunctionCtx() = default;
    ~FunctionCtx();

    // Save tensors needed for the backward pass.
    //
    // Non-output tensors are stored by strong ``shared_ptr<TensorImpl>``.
    // Outputs of this context's own :class:`PythonBackwardNode` are stored as
    // a detached storage alias plus a weak pointer to the live output, a weak
    // pointer to the node, and the output slot index (the reference
    // framework's ``SavedVariable`` for outputs), breaking the
    // ``output -> node -> ctx -> output`` cycle so forward-only calls are
    // freed immediately by refcount without waiting for cyclic GC.
    //
    // Parameters
    // ----------
    // tensors : std::vector<std::shared_ptr<TensorImpl>>
    //     Tensors to save, in the order ``saved_tensors()`` should return
    //     them.  Moved into the context.
    //
    // Notes
    // -----
    // Each call overwrites the previously saved list; saving twice does not
    // accumulate.  To stash non-tensor data (shapes, axes, scalars) use the
    // :func:`store` / :func:`load` extras bag instead.
    void save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors);

    // Record the owning :class:`PythonBackwardNode` so :func:`save_for_backward`
    // can recognise outputs that belong to this node and hold the node weakly.
    void set_owner_node(const std::shared_ptr<Node>& owner);

    // Read back the tensors persisted by :func:`save_for_backward`.
    //
    // Returns
    // -------
    // std::vector<std::shared_ptr<TensorImpl>>
    //     The saved tensors in registration order, re-attaching the owning
    //     node to any saved output whose Python handle has already been
    //     dropped.  Empty when nothing was saved.
    std::vector<std::shared_ptr<TensorImpl>> saved_tensors() const;

    // Visit each owned :class:`TensorImpl` held in ``saved_`` for cyclic GC
    // traversal without constructing temporary ``shared_ptr`` copies.
    void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const;

    bool materialize_grads() const noexcept { return materialize_grads_; }

    // Store an arbitrary Python object under a string key.
    //
    // Used by the pybind11 ``__setattr__`` binding so that Python code
    // assigning ``ctx.shape = x.shape`` lands here.
    //
    // Parameters
    // ----------
    // key : const std::string&
    //     Attribute name.
    // val : py::object
    //     Value to store; ownership is transferred into the extras map.
    void store(const std::string& key, py::object val) {
        if (key == "_materialize_grads") {
            materialize_grads_ = py::cast<bool>(val);
            return;
        }
        extras_[key] = std::move(val);
    }

    // Retrieve a previously stored Python object.
    //
    // Parameters
    // ----------
    // key : const std::string&
    //     Attribute name previously passed to :func:`store`.
    //
    // Returns
    // -------
    // py::object
    //     The stored object, or ``py::none()`` when ``key`` is absent.  The
    //     ``__getattr__`` binding translates the ``None`` sentinel into a
    //     Python ``AttributeError``.
    py::object load(const std::string& key) const {
        auto it = extras_.find(key);
        return (it != extras_.end()) ? it->second : py::none();
    }

    const std::unordered_map<std::string, py::object>& extras() const noexcept { return extras_; }

    void clear() noexcept;

private:
    struct SavedVariable {
        std::shared_ptr<TensorImpl> data;
        std::weak_ptr<TensorImpl> live_output;
        std::weak_ptr<Node> grad_fn;
        std::shared_ptr<VersionCounter> version_sink;
        std::uint32_t output_nr = 0;
        std::int64_t saved_version = 0;
        bool is_output = false;
    };

    std::vector<SavedVariable> saved_;
    std::weak_ptr<Node> owner_node_;
    bool materialize_grads_ = true;
    std::unordered_map<std::string, py::object> extras_;
};

// Autograd graph node that defers backward computation to a Python callable.
//
// When a user subclasses :class:`lucid.autograd.Function` and invokes
// ``MySubclass.apply(...)``, the Python layer constructs a
// ``PythonBackwardNode``, populates :attr:`py_ctx` and :attr:`py_backward_fn`,
// and registers the node as the output tensor's ``grad_fn`` via
// ``_register_python_backward_node``.  The Engine then treats this node like
// any other backward node — when :func:`apply` is called during graph
// traversal, control crosses back into Python with the GIL held.
//
// Attributes
// ----------
// py_ctx : py::object
//     The :class:`FunctionCtx` wrapper carrying saved tensors and extras.
//     Forwarded unchanged as the first argument to ``py_backward_fn``.
// py_backward_fn : py::object
//     The user's ``backward(ctx, grad_output)`` static method.
//     Expected signature: ``(ctx, grad_output: Tensor) -> Tensor |
//     tuple[Tensor | None, ...]``.
// out_shape : Shape
//     Shape of the forward output; used to reconstruct a proper
//     :class:`TensorImpl` wrapper around the raw ``grad_out`` storage.
// out_dtype : Dtype
//     Dtype of the forward output (default ``F32``).
// out_device : Device
//     Device of the forward output (default ``CPU``).
//
// Notes
// -----
// The public ``py_ctx`` / ``py_backward_fn`` data members are exposed
// directly via ``def_readwrite`` so the pybind11 binding can set them
// without accessor boilerplate.
//
// Thread safety: none.  The Engine guarantees single-threaded execution of
// the backward pass, so :func:`apply` does not lock; calling it concurrently
// from multiple threads is undefined.
//
// See Also
// --------
// :class:`lucid.autograd.Function` — Python base class users subclass.
// :class:`FunctionCtx` — context object carrying saved state.
class LUCID_API PythonBackwardNode : public Node, public GraphBarrier {
public:
    PythonBackwardNode() = default;
    ~PythonBackwardNode() override;

    // The FunctionCtx object forwarded to py_backward_fn as its first argument.
    py::object py_ctx;

    // The Python callable implementing the backward pass.
    // Signature: (ctx: FunctionCtx, grad_output: Tensor) -> Tensor | tuple[Tensor]
    py::object py_backward_fn;

    // Set for a Function whose backward is not itself differentiable — it
    // leaves the graph (numpy, detach, a closed-form factor it does not
    // record).  create_graph=True through it is refused rather than given
    // a gradient with no path back, which a second derivative would read
    // as zero.  The reference framework's ``once_differentiable``.
    bool once_differentiable = false;

    // Metadata of the forward output tensor — used to reconstruct a
    // TensorImpl wrapper around the incoming gradient Storage.
    Shape out_shape;
    Dtype out_dtype = Dtype::F32;
    Device out_device = Device::CPU;

    // Metadata of every forward output, when ``forward`` returned more than
    // one tensor.  Empty for the single-output case, which keeps the
    // original non-barrier path untouched.
    //
    // A ``Function`` returning a tuple produces several tensors sharing this
    // one node, distinguished by their ``grad_output_nr``.  Their gradients
    // arrive independently and at different times during the traversal, so
    // the node becomes a barrier: it parks each arriving gradient in
    // :attr:`slots_` and runs the user's ``backward`` once, with the
    // full set, when the engine reaches it.  The topological order
    // guarantees every contributor has already run by then.
    std::vector<Shape> out_shapes;
    std::vector<Dtype> out_dtypes;
    std::vector<Device> out_devices;

    // Human-readable node name for debugger / profiler output.
    //
    // Returns
    // -------
    // std::string_view
    //     The literal ``"PythonBackward"``.
    std::string_view name() const noexcept { return "PythonBackward"; }

    // Invoke the Python backward callable and collect the resulting gradients.
    //
    // Acquires the GIL, wraps ``grad_out`` in a temporary :class:`TensorImpl`
    // matching :attr:`out_shape` / :attr:`out_dtype` / :attr:`out_device`,
    // calls ``py_backward_fn(py_ctx, grad_tensor)``, then unpacks the
    // returned tuple/list/tensor back into a vector of :class:`Storage`
    // values — one per forward input.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient delivered by the Engine; assumed to match the
    //     output dtype/device of the original forward call.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One :class:`Storage` per input edge.  Python ``None`` entries
    //     become empty ``CpuStorage{}`` values representing "no gradient
    //     for this input".
    //
    // Raises
    // ------
    // LucidError
    //     The Python ``backward_fn`` is unset.
    // py::error_already_set
    //     ``backward_fn`` raised.  It is not rewrapped: the binding that
    //     started the backward pass restores the original Python exception,
    //     so the caller sees its own type, message and traceback.
    std::vector<Storage> apply(Storage grad_out) override;

    // create_graph=True: the user's backward, run on the live gradient.
    //
    // The same Python ``backward`` is called, but handed the upstream
    // gradient as the tensor the engine holds — ``grad_fn`` and all — and
    // its results are returned as tensors rather than storages.  Written in
    // Lucid ops, as a backward is, it records its own graph the way any
    // other code does when grad mode is on, so a custom Function
    // differentiates twice without saying anything about it.  The engine
    // calls it for a single-output Function; one with several outputs is a
    // barrier and collects its gradients per output
    // (:meth:`apply_barrier_for_graph`).
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Whether this node stands behind several forward outputs.
    //
    // Returns
    // -------
    // bool
    //     ``true`` once :attr:`out_shapes` describes more than one output,
    //     which routes the engine through the barrier protocol.  A
    //     single-output custom op stays on the direct path, exactly as
    //     before multi-output support existed.
    bool is_barrier() const noexcept override { return out_shapes.size() > 1; }
    // A Python ``None`` comes back as an empty storage.
    bool empty_grad_is_none() const noexcept override { return true; }

    // Park one output's gradient until the rest arrive.
    //
    // Parameters
    // ----------
    // input_nr : std::uint32_t
    //     Which forward output this gradient belongs to.  Out-of-range
    //     values are ignored rather than throwing, matching the other
    //     barrier nodes and keeping the engine robust against pruned graphs.
    // grad : Storage
    //     The arriving gradient.  A slot reached by more than one path
    //     accumulates rather than overwrites.
    void accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) override;

    // Run the user's ``backward`` once with every output's gradient.
    //
    // Outputs that never received a gradient — because the loss did not
    // depend on them — are materialised as zeros of the right shape, dtype
    // and device, so ``backward`` always sees one real tensor per output
    // and never has to guard against ``None``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One :class:`Storage` per forward input, as for :func:`apply`.
    std::vector<Storage> apply_barrier() override;

    // The create_graph half of the barrier: an output's gradient is parked
    // as the tensor it arrived as, and ``backward`` runs once on them all,
    // as :meth:`apply_barrier` runs it on the eager ones.
    void accumulate_barrier_grad_for_graph(std::uint32_t input_nr, TensorImplPtr grad) override;
    std::vector<TensorImplPtr> apply_barrier_for_graph() override;

    void release_saved() override;

    // The create_graph gradients parked for the running pass.  What
    // ``backward`` saved belongs to :attr:`py_ctx`, whose own traversal
    // reaches it.
    void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const override;

private:
    // One slot per forward output, engaged once a gradient has arrived — a
    // buffer in an eager pass, a tensor under create_graph.
    struct Slots {
        std::vector<std::optional<Storage>> eager;
        std::vector<TensorImplPtr> graph;
    };

    // Make the slots the running pass's (GraphBarrier.h).
    void enter_pass();

    // The user's ``backward`` result under create_graph, as tensors.
    std::vector<TensorImplPtr> graph_results(const py::object& result) const;

    Slots slots_;
    PassSlots<Slots> passes_;

    // Shared tail of ``apply`` and ``apply_barrier``: call into Python with
    // the prepared gradient tensors and unpack the returned per-input
    // gradients.
    std::vector<Storage> invoke_backward(const py::tuple& grads);
};

// Register the custom-function bindings into a pybind11 module.
//
// Exposes three Python-visible names under the receiving module:
//
//   - ``FunctionCtx`` — opaque shared-pointer class with
//     ``save_for_backward(*tensors)``, the ``saved_tensors`` property,
//     and permissive ``__setattr__`` / ``__getattr__`` for extras.
//   - ``_PythonBackwardNode`` — internal :class:`Node` subclass with
//     read/write ``ctx`` and ``backward_fn`` fields.
//   - ``_register_python_backward_node(output, node, inputs)`` — wires
//     ``node`` into the autograd graph as ``output``'s ``grad_fn``,
//     materialising :class:`AccumulateGrad` nodes for any leaves that
//     need them and recording version counters for in-place mutation
//     checks.
//
// Parameters
// ----------
// m : py::module_&
//     The pybind11 module to register into.  Called once during engine
//     module initialisation.
//
// See Also
// --------
// :mod:`lucid.autograd._python_node` — Python-side caller of
// ``_register_python_backward_node``.
void register_custom_function(py::module_& m);

}  // namespace lucid
