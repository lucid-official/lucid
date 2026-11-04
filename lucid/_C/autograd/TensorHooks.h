// lucid/_C/autograd/TensorHooks.h
//
// Tensor gradient hooks and ``retain_grad``, run by the engine where the
// tensor's gradient is complete.
//
// A non-leaf tensor's gradient is the sum of everything that reaches its
// producer's output slot ``(grad_fn, grad_output_nr)``, so its hooks live on
// that node, slot by slot — the node holds them strongly, since the tensor
// itself is often gone by the time backward runs (consumers keep only weak
// handles on their inputs).  The engine runs them once the slot's gradient
// has arrived, before the node does, and what they return is what flows on.
// ``retain_grad`` is the same kind of slot entry, run after the hooks, so a
// retained gradient is the hooked one.
//
// A leaf's hooks live in its :class:`AutogradMeta` instead: its
// :class:`AccumulateGrad` is made lazily and dropped by ``release_root`` and
// ``detach_``, so a hook stored there would be lost.  They run before the
// gradient is accumulated into ``.grad``.
//
// What a hook *is* — the list of callables, their order, their removal — is
// Python's business.  Each slot holds one Python ``runner(grad) -> grad |
// None`` that Python installs once, and the engine calls only that.  A slot
// that was never asked for does not exist, so a backward pass with no hooks
// never touches Python.
//
// This file also owns what every gradient Python hands back has to pass
// before the engine uses it — a hook's, a module hook's, a custom Function's,
// ``Tensor.grad =``'s: one converter from Python and one check against the
// slot the gradient replaces (``tensor_from_python``, ``check_grad_kind``,
// ``grad_slot_buffer`` below).

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/ErrorBuilder.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Node.h"

namespace py = pybind11;

namespace lucid {

// One tensor's hooks and retained-gradient target.
//
// Attributes
// ----------
// runner : py::object
//     The Python callable that runs the tensor's hooks, or a null handle
//     while it has none (a slot made by ``retain_grad`` alone).
// retain : std::weak_ptr<TensorImpl>
//     The non-leaf tensor whose ``.grad`` keeps this slot's gradient, when
//     ``retain_grad`` asked for it.  Weak: the slot must not keep it alive.
// shape : Shape
//     The tensor's shape, to give a gradient Storage a shape again.
//
// Notes
// -----
// The runner is dropped under the GIL, wherever the last owner lets go of the
// slot — a node or a tensor can die on a thread that does not hold it.
struct LUCID_API TensorHookSlot {
    py::object runner;
    py::object post_runner;
    std::weak_ptr<TensorImpl> retain;
    Shape shape;

    TensorHookSlot() = default;
    ~TensorHookSlot();

    TensorHookSlot(const TensorHookSlot&) = delete;
    TensorHookSlot& operator=(const TensorHookSlot&) = delete;

    void clear() noexcept;
};

// The hook slots of one node, by output slot, plus node-level backward hooks
// and packed saved tensors.
//
// Slots are held through ``shared_ptr`` so that the engine can keep the one it
// is running alive: a hook may register another one (growing this list) or run
// a backward pass of its own that releases this node's hooks.
class LUCID_API NodeHooks {
public:
    NodeHooks() = default;
    ~NodeHooks();

    NodeHooks(const NodeHooks&) = delete;
    NodeHooks& operator=(const NodeHooks&) = delete;

    // The slot of output ``output_nr``, or null when nothing was registered
    // there.
    std::shared_ptr<TensorHookSlot> find(std::uint32_t output_nr) const;

    // The slot of output ``output_nr``, made empty if it does not exist yet.
    std::shared_ptr<TensorHookSlot> ensure(std::uint32_t output_nr);

    // Every registered slot, for CPython cyclic GC traversal.
    const std::vector<std::shared_ptr<TensorHookSlot>>& slots() const noexcept { return slots_; }

    void clear() noexcept;

    py::object node_pre_runner;
    py::object node_post_runner;
    std::vector<py::object> packed_inputs;
    py::object packed_output;
    bool saved_packed = false;

private:
    std::vector<std::shared_ptr<TensorHookSlot>> slots_;
};

// Whether output ``output_nr`` of ``node`` has a hook slot.  For a node with
// no hooks this is a null test and nothing else.
inline bool has_slot_hooks(const Node& node, std::uint32_t output_nr) {
    const NodeHooks* hooks = node.tensor_hooks();
    return hooks != nullptr && hooks->find(output_nr) != nullptr;
}

inline bool has_node_pre_hooks(const Node& node) {
    const NodeHooks* hooks = node.tensor_hooks();
    return hooks != nullptr && static_cast<bool>(hooks->node_pre_runner);
}

inline bool has_node_post_hooks(const Node& node) {
    const NodeHooks* hooks = node.tensor_hooks();
    return hooks != nullptr && static_cast<bool>(hooks->node_post_runner);
}

// ── Running them ─────────────────────────────────────────────────────────────
//
// One pair of helpers for every traversal: eager backward and ``grad()`` hand
// over Storage, their ``create_graph`` forms hand over tensors.  Each returns
// the gradient that flows on.

// Run the hooks of output ``output_nr`` of ``producer`` on that slot's whole
// gradient, then retain the result if the tensor asked for it.
//
// The hooks run with the GIL held and grad mode off.  A hook that returned a
// tensor replaces the gradient — after the checks ``Tensor.grad =`` makes —
// and one that returned ``None`` leaves it as the hook left it, in-place
// writes included.  Once a hook has run, a CPU gradient is copied before it
// flows on: the hook may keep what it was handed, or hand back a tensor
// somebody else reads, and the engine adds into gradients in place.
//
// Raises
// ------
// py::error_already_set
//     A hook raised; the exception reaches the caller of backward as it was.
// py::type_error
//     A hook returned something that is neither a tensor nor ``None``.
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     A hook returned a gradient of another kind than it was given.
LUCID_API Storage run_slot_hooks(Node& producer, std::uint32_t output_nr, Storage grad);

// :func:`run_slot_hooks` for ``create_graph``: the hooks run with grad mode on
// and see the gradient with its graph, and what they return keeps its own.
LUCID_API TensorImplPtr run_slot_hooks_for_graph(Node& producer,
                                                 std::uint32_t output_nr,
                                                 TensorImplPtr grad);

// Run a leaf's hooks on the gradient about to be accumulated into its
// ``.grad`` (already in the leaf's dtype).  Same contract as
// :func:`run_slot_hooks`, without the retain step a leaf does not have.
LUCID_API Storage run_leaf_hooks(const TensorImpl& leaf, Storage grad);

// :func:`run_leaf_hooks` for ``create_graph``.
LUCID_API TensorImplPtr run_leaf_hooks_for_graph(const TensorImpl& leaf, TensorImplPtr grad);

// Run a leaf's post-accumulate-grad hooks after its ``.grad`` has been updated.
LUCID_API void run_leaf_post_hooks(const TensorImpl& leaf, bool graph);

// Run a node's registered prehooks on its incoming output gradient before
// ``apply`` runs.
LUCID_API Storage run_node_pre_hooks(Node& node, Storage grad_out);
LUCID_API TensorImplPtr run_node_pre_hooks_for_graph(Node& node, TensorImplPtr grad_out);

// Run a node's registered post-hooks on ``(grad_inputs, grad_outputs)`` after
// ``apply`` returns.
LUCID_API void
run_node_post_hooks(Node& node, std::vector<Storage>& grad_inputs, const Storage& grad_out);
LUCID_API void run_node_post_hooks_for_graph(Node& node,
                                             std::vector<TensorImplPtr>& grad_inputs,
                                             const TensorImplPtr& grad_out);

// ── Registering them ─────────────────────────────────────────────────────────

// The runner of ``t``'s hooks, installed from ``make()`` the first time.
//
// A leaf's slot is its own; a non-leaf's is its producer's output slot, so
// every consumer — made before the hook or after — sends its gradient through
// it.  ``t``'s shape is recorded with it.
//
// Notes
// -----
// A non-leaf's runner lives on its producer for as long as the node does.
// The engine drops it once ``backward(retain_graph=False)`` has released the
// node's saved state (:meth:`Node::release_tensor_hooks`) — the node can never
// run again, and dropping it breaks the cycle tensor → node → runner → hook
// closure → tensor that Python's collector cannot see through the engine.
// Two consequences the caller should know:
//
// * a hook registered on the output of a node that was *already* released
//   that way is installed but never fires (any further backward through the
//   node is refused), and it keeps that cycle until the node itself goes;
// * a node that saved nothing for backward is never released, so a hook on
//   its output that refers to the output keeps the cycle while the graph
//   lives — as does a hook on a graph that never runs backward at all.
//
// A leaf's runner lives in its own AutogradMeta, so a leaf hook that refers
// to the leaf keeps the leaf alive the same way.  The Python side
// (CHA-151-A) should hold the tensor weakly in its runner.
//
// Raises
// ------
// LucidError
//     ``t`` does not require grad, so no gradient will ever reach it.
LUCID_API py::object tensor_hook_runner(const TensorImplPtr& t, const py::object& make);

// Whether ``t``'s slot has a runner installed.
LUCID_API bool has_tensor_hooks(const TensorImpl& t);

// The runner of ``t``'s post-accumulate-grad hooks (leaf tensors only).
LUCID_API py::object tensor_post_accumulate_grad_hook_runner(const TensorImplPtr& t,
                                                             const py::object& make);

// Whether ``t`` has a post-accumulate-grad hook runner installed.
LUCID_API bool has_post_accumulate_grad_hooks(const TensorImpl& t);

// The runner of ``node``'s prehooks or post-hooks, installed from ``make()``
// the first time.
LUCID_API py::object node_prehook_runner(Node& node, const py::object& make);
LUCID_API py::object node_hook_runner(Node& node, const py::object& make);

// ── Saved-tensor hooks for built-in ops ──────────────────────────────────────

LUCID_API void push_saved_tensors_hooks(py::object pack_hook, py::object unpack_hook);
LUCID_API void pop_saved_tensors_hooks() noexcept;
LUCID_API bool has_saved_tensors_hooks() noexcept;
LUCID_API py::object pack_saved_tensor(const TensorImplPtr& t);
LUCID_API TensorImplPtr unpack_saved_tensor(const py::object& entry);

// Keep ``t``'s gradient in its ``.grad`` during backward, even though it is not
// a leaf.  Sets the flag and, for a non-leaf, registers ``t`` at its producer's
// output slot — called again after an in-place op moves ``t`` to a new one.
// A leaf keeps its gradient anyway; only the flag changes.
LUCID_API void retain_grad(const TensorImplPtr& t);

// ── Gradients handed back by Python ──────────────────────────────────────────
//
// Every gradient Python hands back to the engine crosses one of these before
// it reaches a kernel or an accumulator: a tensor hook's result, a module
// backward hook's tuple (ModuleHookNode.cpp), a custom Function's backward
// (CustomFunction.cpp — its shape, dtype and device are held to the input by
// ``lucid.autograd._python_node._validate`` first) and ``Tensor.grad =``.  A
// gradient slot is read as the tensor it stands for, so a value of another
// kind is not converted but refused: read as it was, a short buffer was read
// past its end and a float16 one had its bits taken for float32.

// The tensor a Python value holds — a ``TensorImpl``, or a ``lucid.Tensor``
// through its ``impl`` — or null for ``None``.  The one Python → TensorImpl
// conversion of the autograd layer.
//
// Raises
// ------
// py::type_error
//     ``obj`` is anything else; the message names ``what``, the value's role
//     ("a tensor hook's result", ...).
LUCID_API TensorImplPtr tensor_from_python(py::handle obj, const std::string& what);

// Refuse ``g`` unless it has the dtype, device and shape of the gradient slot
// it replaces — checked in that order, as the reference checks them.
// ``what`` names ``g`` in the message.
//
// Raises
// ------
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     ``g`` is of another kind.
LUCID_API void check_grad_kind(const ErrorBuilder& err,
                               const std::string& what,
                               Dtype dtype,
                               Device device,
                               const Shape& shape,
                               const TensorImpl& g);

// ``g``'s values as a buffer read from its first byte in row-major order —
// the form a gradient slot holds.  A CPU view answers ``storage()`` with a
// packed copy of the elements it reads; a Metal tensor's array is its
// elements, unless it is a window at an offset or with strides into another
// tensor's array, which only the private ``_make_view`` builds and which has
// no such buffer to hand over.
//
// Raises
// ------
// NotImplementedError
//     ``g`` is such a Metal window.
LUCID_API Storage grad_slot_buffer(const ErrorBuilder& err, const TensorImpl& g);

// ── Shared with the rest of the engine ───────────────────────────────────────

// A gradient buffer its holder alone owns, for a holder that keeps adding
// into it.  A metal add replaces the array rather than writing into it, so a
// metal gradient comes back as it is.
LUCID_API Storage own_grad_copy(const Storage& s);

// ``s``, a gradient a node that calls into Python hands on, told apart from
// that node's ``None``.  Those nodes say ``None`` with an empty storage
// (``Node::empty_grad_is_none``), and a CPU gradient of zero elements owns
// no memory either: it read as ``None``, and its input's ``.grad`` stayed
// unset where the reference gives an empty gradient.  Such a buffer is
// given a placeholder pointer, which its zero bytes never read.
LUCID_API Storage present_gradient(Storage s);

// The buffer ``self.grad = g`` installs.
//
// The gradient slot is a bare Storage, read as ``self``'s dtype, device and
// shape; ``g`` must have all three — checked in that order, as the reference
// does — and may not be ``self``.  A hook's returned gradient passes the same
// checks against the gradient it replaces.
//
// Raises
// ------
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     ``g`` is of another kind.
// LucidError
//     ``g`` is ``self``, or a Metal strided view with no buffer of its own.
LUCID_API Storage assignable_grad(const TensorImpl& self, const TensorImpl& g);

// ── Python reference lifetime & cyclic garbage collection ────────────────────

// Release ``obj``'s reference under the GIL if the interpreter is still
// running.  Detaches the underlying ``PyObject*`` before ``Py_DECREF`` so
// re-entrant Python finalizers cannot observe a half-cleared handle.
LUCID_API void release_py_object(py::object& obj) noexcept;

// Enable CPython cyclic garbage collection (``Py_TPFLAGS_HAVE_GC``,
// ``tp_traverse``, ``tp_clear``) on pybind11 heap types that can participate
// in reference cycles through the C++ autograd graph.
LUCID_API void enable_tensor_autograd_gc(PyHeapTypeObject* heap_type);
LUCID_API void enable_node_autograd_gc(PyHeapTypeObject* heap_type);
LUCID_API void enable_function_ctx_gc(PyHeapTypeObject* heap_type);
LUCID_API void enable_module_hook_state_gc(PyHeapTypeObject* heap_type);

}  // namespace lucid
