// lucid/_C/autograd/TensorHooks.cpp
//
// The hook slots, the two helpers every traversal runs them through (one for
// Storage gradients, one for the tensors of a create_graph pass), and their
// registration.

#include "TensorHooks.h"

#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/PhantomMode.h"
#include "AccumulateGrad.h"
#include "CustomFunction.h"
#include "Helpers.h"
#include "ModuleHookNode.h"

namespace lucid {

namespace {

constexpr const char* kHookResult = "a hook's returned gradient";

// Call the slot's runner on ``grad`` with grad mode ``graph`` — on for a
// create_graph pass, so what a hook computes joins the graph, off otherwise.
TensorImplPtr call_runner(const TensorHookSlot& slot, const TensorImplPtr& grad, bool graph) {
    py::gil_scoped_acquire gil;
    // A reference of its own: the hook may let go of the slot.
    const py::object runner = slot.runner;
    py::object result;
    {
        const GradModeGuard mode(graph);
        result = runner(grad);
    }
    return tensor_from_python(result, "a tensor hook's result");
}

// The one place an eager gradient goes through a runner.
Storage run_hooks(const TensorHookSlot& slot, Storage grad, const Shape& shape) {
    if (auto* cs = std::get_if<CpuStorage>(&grad); cs && cs->ptr.use_count() > 1)
        grad = own_grad_copy(grad);
    const Dtype dtype = storage_dtype(grad);
    const Device device = storage_device(grad);
    auto given = std::make_shared<TensorImpl>(std::move(grad), shape, dtype, device, false);
    const TensorImplPtr returned = call_runner(slot, given, /*graph=*/false);
    Storage out;
    if (!returned || returned == given) {
        // ``None``: the gradient as the hook left it, an in-place write into
        // the tensor it was handed included.
        out = given->storage();
    } else {
        const ErrorBuilder err("Tensor.register_hook");
        check_grad_kind(err, kHookResult, dtype, device, shape, *returned);
        out = grad_slot_buffer(err, *returned);
    }
    return own_grad_copy(out);
}

// The one place a create_graph gradient goes through a runner.
TensorImplPtr run_hooks_for_graph(const TensorHookSlot& slot, TensorImplPtr grad) {
    const TensorImplPtr returned = call_runner(slot, grad, /*graph=*/true);
    if (!returned || returned == grad)
        return grad;
    check_grad_kind(ErrorBuilder("Tensor.register_hook"), kHookResult, grad->dtype(),
                    grad->device(), grad->shape(), *returned);
    return returned;
}

// The tensor that asked ``producer``'s output ``output_nr`` to keep its
// gradient, if it is alive and still sits there.  An in-place op moves a
// tensor to a new slot and registers it there; the old slot's entry then
// stops matching.
TensorImplPtr
retaining_tensor(const TensorHookSlot& slot, const Node& producer, std::uint32_t output_nr) {
    TensorImplPtr t = slot.retain.lock();
    if (!t || !t->retains_grad() || t->is_leaf() || t->grad_fn().get() != &producer ||
        t->grad_output_nr() != output_nr)
        return nullptr;
    return t;
}

}  // namespace

// ── Slots ────────────────────────────────────────────────────────────────────

TensorHookSlot::~TensorHookSlot() {
    clear();
}

void TensorHookSlot::clear() noexcept {
    retain.reset();
    release_py_object(runner);
}

std::shared_ptr<TensorHookSlot> NodeHooks::find(std::uint32_t output_nr) const {
    return output_nr < slots_.size() ? slots_[output_nr] : nullptr;
}

std::shared_ptr<TensorHookSlot> NodeHooks::ensure(std::uint32_t output_nr) {
    if (output_nr >= slots_.size())
        slots_.resize(static_cast<std::size_t>(output_nr) + 1);
    auto& slot = slots_[output_nr];
    if (!slot)
        slot = std::make_shared<TensorHookSlot>();
    return slot;
}

// ── Running ──────────────────────────────────────────────────────────────────

Storage run_slot_hooks(Node& producer, std::uint32_t output_nr, Storage grad) {
    const NodeHooks* hooks = producer.tensor_hooks();
    if (hooks == nullptr)
        return grad;
    // Held for the whole call: the hook may release or grow the node's slots.
    const std::shared_ptr<TensorHookSlot> slot = hooks->find(output_nr);
    if (!slot)
        return grad;
    if (slot->runner)
        grad = run_hooks(*slot, std::move(grad), slot->shape);
    if (const TensorImplPtr t = retaining_tensor(*slot, producer, output_nr);
        t && !t->absorbs_phantom_gradient(grad)) {
        if (t->grad_as_impl()) {
            auto g_impl = std::make_shared<TensorImpl>(own_grad_copy(grad), t->shape(), t->dtype(),
                                                       t->device(), false);
            t->accumulate_grad_impl(std::move(g_impl));
        } else {
            // A buffer of its own: what flows on is added into in place.
            auto& kept = t->mutable_grad_storage();
            if (!kept.has_value())
                kept = own_grad_copy(grad);
            else
                accumulate_into(*kept, grad);
        }
    }
    return grad;
}

TensorImplPtr
run_slot_hooks_for_graph(Node& producer, std::uint32_t output_nr, TensorImplPtr grad) {
    const NodeHooks* hooks = producer.tensor_hooks();
    if (hooks == nullptr || !grad)
        return grad;
    const std::shared_ptr<TensorHookSlot> slot = hooks->find(output_nr);
    if (!slot)
        return grad;
    if (slot->runner)
        grad = run_hooks_for_graph(*slot, std::move(grad));
    // Summed by add_op, so the retained gradient carries its graph as a
    // leaf's does.
    if (const TensorImplPtr t = retaining_tensor(*slot, producer, output_nr))
        t->accumulate_grad_impl(grad);
    return grad;
}

Storage run_leaf_hooks(const TensorImpl& leaf, Storage grad) {
    const std::shared_ptr<TensorHookSlot> slot = leaf.leaf_hooks();
    if (!slot || !slot->runner)
        return grad;
    return run_hooks(*slot, std::move(grad), leaf.shape());
}

TensorImplPtr run_leaf_hooks_for_graph(const TensorImpl& leaf, TensorImplPtr grad) {
    const std::shared_ptr<TensorHookSlot> slot = leaf.leaf_hooks();
    if (!slot || !slot->runner || !grad)
        return grad;
    return run_hooks_for_graph(*slot, std::move(grad));
}

// ── Registering ──────────────────────────────────────────────────────────────

py::object tensor_hook_runner(const TensorImplPtr& t, const py::object& make) {
    const ErrorBuilder err("Tensor.register_hook");
    if (!t)
        err.invalid_argument("expected a tensor");
    if (!t->requires_grad())
        err.fail("cannot register a hook on a tensor that does not require grad");
    std::shared_ptr<TensorHookSlot> slot;
    if (t->is_leaf()) {
        auto& own = t->mutable_leaf_hooks();
        if (!own)
            own = std::make_shared<TensorHookSlot>();
        slot = own;
    } else {
        if (!t->grad_fn())
            err.fail("a non-leaf tensor without a grad_fn has no gradient to hook");
        slot = t->grad_fn()->ensure_tensor_hooks().ensure(t->grad_output_nr());
    }
    if (!slot->runner) {
        py::object runner = make();
        if (runner.is_none())
            err.invalid_argument("the runner factory returned None");
        slot->runner = std::move(runner);
        slot->shape = t->shape();
    }
    return slot->runner;
}

bool has_tensor_hooks(const TensorImpl& t) {
    if (t.is_leaf()) {
        const auto& slot = t.leaf_hooks();
        return slot && slot->runner;
    }
    const auto& fn = t.grad_fn();
    if (!fn || fn->tensor_hooks() == nullptr)
        return false;
    const auto slot = fn->tensor_hooks()->find(t.grad_output_nr());
    return slot && slot->runner;
}

void retain_grad(const TensorImplPtr& t) {
    if (!t)
        return;
    t->set_retain_grad(true);
    if (t->is_leaf() || !t->grad_fn())
        return;
    const auto slot = t->grad_fn()->ensure_tensor_hooks().ensure(t->grad_output_nr());
    slot->retain = t;
    slot->shape = t->shape();
}

// ── Gradients handed back by Python ──────────────────────────────────────────

TensorImplPtr tensor_from_python(py::handle obj, const std::string& what) {
    if (obj.is_none())
        return nullptr;
    if (py::isinstance<TensorImpl>(obj))
        return obj.cast<TensorImplPtr>();
    if (py::hasattr(obj, "impl")) {
        const py::object impl = obj.attr("impl");
        if (py::isinstance<TensorImpl>(impl))
            return impl.cast<TensorImplPtr>();
    }
    throw py::type_error(what + " must be a tensor or None, got " +
                         std::string(py::str(py::type::of(obj).attr("__name__"))));
}

void check_grad_kind(const ErrorBuilder& err,
                     const std::string& what,
                     Dtype dtype,
                     Device device,
                     const Shape& shape,
                     const TensorImpl& g) {
    if (g.dtype() != dtype)
        err.dtype_mismatch(dtype, g.dtype(), what + " must have the tensor's dtype");
    if (g.device() != device)
        err.device_mismatch(device, g.device(), what + " must be on the tensor's device");
    if (g.shape() != shape)
        err.shape_mismatch(shape, g.shape(), what + " must have the tensor's shape");
}

Storage grad_slot_buffer(const ErrorBuilder& err, const TensorImpl& g) {
    if (!storage_is_cpu(g.raw_storage()) && g.family_key() == nullptr &&
        (g.storage_offset() != 0 || !g.is_contiguous()))
        err.not_implemented("a Metal gradient that is a strided view of another tensor's array — "
                            "pass a contiguous copy");
    return TensorImpl::detach_gpu_grad_slot(g.storage());
}

// ── Shared ───────────────────────────────────────────────────────────────────

Storage own_grad_copy(const Storage& s) {
    if (!storage_is_cpu(s))
        return s;
    const Dtype dt = storage_dtype(s);
    return clone_storage(s, storage_nbytes(s) / dtype_size(dt), dt, Device::CPU);
}

Storage present_gradient(Storage s) {
    auto* cpu = std::get_if<CpuStorage>(&s);
    if (cpu == nullptr || cpu->ptr || cpu->nbytes != 0)
        return s;
    static const std::shared_ptr<std::byte[]> kPlaceholder(new std::byte[1]{});
    cpu->ptr = kPlaceholder;
    return s;
}

// Every write of an assigned gradient lands here or in a hook's replacement:
// ``.grad =`` and hook returns.  The slot is a bare Storage — no dtype, shape
// or device of its own — and ``.grad``, the optimizers and the next backward
// read it as ``self``'s.  A buffer of any
// other kind was read as if it were one: float32 bits taken for float16
// values (``[1, 2, 4]`` came back ``[0, 1.875, 1.875]``), three elements read
// out of a buffer of two, a Metal array labelled CPU.  The reference refuses
// each of these, in this order, and so does this, before the slot changes.
Storage assignable_grad(const TensorImpl& self, const TensorImpl& g) {
    const ErrorBuilder err("Tensor.grad");
    if (&g == &self)
        err.fail("a tensor cannot be assigned as its own gradient");
    check_grad_kind(err, "an assigned gradient", self.dtype(), self.device(), self.shape(), g);
    // The phantom write rule (core/PhantomMode.h): a phantom's slot takes
    // any gradient and holds a phantom of its own geometry; a real one
    // refuses a phantom, whose buffer it would otherwise hold.
    if (absorbs_phantom_write(self.raw_storage(), g.raw_storage(), "Tensor.grad"))
        return phantom_storage(self.shape(), self.dtype(), self.device());
    if (storage_is_cpu(g.raw_storage()) && g.is_contiguous() && !g.is_dense()) {
        const auto& cpu = std::get<CpuStorage>(g.raw_storage());
        const std::size_t off = g.storage_offset();
        std::shared_ptr<std::byte[]> view_ptr(cpu.ptr, cpu.ptr.get() + off);
        return TensorImpl::detach_gpu_grad_slot(Storage{CpuStorage{
            std::move(view_ptr), g.nbytes(), g.dtype(), cpu.version, cpu.offset_in_allocation + off,
            cpu.allocation_nbytes != 0 ? cpu.allocation_nbytes : cpu.nbytes}});
    }
    return grad_slot_buffer(err, g);
}

// ── Python reference lifetime & cyclic garbage collection ────────────────────
//
// What a wrapper may tell the collector, and what it may clear (LCD-459).
//
// A C++ object behind a Python wrapper can have other C++ owners the
// collector cannot see: a tensor saved by a later op, a node on another
// tensor's edges, a hook state shared by a module call's input and output
// nodes.  Reporting a Python object such an object holds as the *wrapper's*
// reference is a lie whenever another owner is alive.  The collector then
// subtracts a reference that a live path still holds, finds a live object
// unreachable, and clears it.  LCD-392's first hooks walked the whole graph
// reachable from every wrapper, and so did all of this:
//   * ``tp_clear`` on a garbage wrapper cut ``grad_fn`` off a TensorImpl that a
//     live graph had saved, so a second derivative came back ``[2, 2, 2]``
//     where the reference gives ``[10, 10, 10]``;
//   * every wrapper re-walked the graph behind it, so a collection over 4,000
//     chained tensors took 8 s.
//
// A wrapper therefore reports only its *exclusive region*.  The region starts
// with the object it wraps, if the wrapper's holder is that object's only
// owner.  An object reached from a member joins once every owner it has
// (its ``use_count``) is accounted for by an edge from a member.  Only Python
// references held by members are visited.  This is the reference framework's
// rule — a tensor wrapper traverses into its ``grad_fn`` only as the sole
// owner of both — carried along the graph, so a cycle through a module's
// hooked output a few ops deep is still collected (LCD-392).
//
//   * Sound: a member is reachable only through the wrapper, so when the
//     wrapper is garbage, so is every reference a member holds.
//   * Visited once: regions of different wrappers are disjoint (a root's only
//     owner is its wrapper, never a member), so each ``py::object`` is
//     reported at most once per pass and ``gc_refs`` cannot underflow.
//   * Linear: an object is explored only from the one region that owns it,
//     and a shared node stops the walk at the first edge into it.
//
// An object that two regions own jointly — a node on both a kept tensor's
// path and a kept output's — belongs to neither and is not collected through
// either wrapper.  That is the price of never clearing a live object.
//
// ``tp_clear`` mutates the wrapped object only when the wrapper is still its
// sole owner, and then only drops references: the members die because the
// wrapper's object was their only way in.

namespace {

// ``grad_as_impl`` hands back a copy, which is one owner more than the
// tensor's own; the region subtracts it.  If the accessor ever returns a
// reference, that correction would let a shared gradient join, so pin it.
static_assert(!std::is_reference_v<decltype(std::declval<const TensorImpl&>().grad_as_impl())>,
              "grad_as_impl() returns a reference now: drop the copy correction in drain_tensors");

// A pybind11 instance's C++ ``T`` and the ``use_count`` of the holder it
// carries — every class this file hooks is held by ``std::shared_ptr`` —
// or ``{nullptr, 0}``.  Runs inside ``tp_traverse`` / ``tp_clear``, so it
// neither raises nor allocates.
//
// pybind11 keeps one value slot per registered C++ type the Python type
// derives from, keyed by the most-derived one: an ``AccumulateGrad`` or
// ``_PythonBackwardNode`` instance holds its value under its own type_info,
// never under ``Node``'s.  ``get_value_and_holder(tinfo, false)`` therefore
// answers "not found" for every registered subclass, with an empty
// value_and_holder whose ``operator bool`` reads through a null pointer —
// one garbage collection with such a node alive was a SIGSEGV.  So walk the
// slots and take the one whose type is ``T`` or a subclass of it.  The
// subclass pointer is a ``T*``, and its ``shared_ptr`` holder reads as a
// ``shared_ptr<T>``, only when no registered type in the hierarchy uses
// multiple inheritance (``simple_type``) — the condition pybind11 itself
// casts under — and anything else is declined rather than guessed.
template <typename T>
struct Wrapped {
    T* ptr = nullptr;
    long owners = 0;
};

template <typename T>
Wrapped<T> wrapped(PyObject* self) noexcept {
    if (self == nullptr)
        return {};
    const auto* tinfo = py::detail::get_type_info(typeid(T));
    if (tinfo == nullptr)
        return {};
    PyTypeObject* type = Py_TYPE(self);
    if (type != tinfo->type && !PyType_IsSubtype(type, tinfo->type))
        return {};
    auto* inst = reinterpret_cast<py::detail::instance*>(self);
    for (const auto& vh : py::detail::values_and_holders(inst)) {
        const bool exact = vh.type == tinfo;
        if (!exact && !(tinfo->simple_type && PyType_IsSubtype(vh.type->type, tinfo->type)))
            continue;
        if (!vh.holder_constructed() || vh.value_ptr() == nullptr)
            return {};
        return {static_cast<T*>(vh.value_ptr()), vh.holder<std::shared_ptr<T>>().use_count()};
    }
    return {};
}

// The wrapped object, when the wrapper is its only owner.
template <typename T>
T* solely_owned(PyObject* self) noexcept {
    const Wrapped<T> w = wrapped<T>(self);
    return w.owners == 1 ? w.ptr : nullptr;
}

// One wrapper's exclusive region, and the visits it owes the collector.
class ExclusiveRegion {
public:
    ExclusiveRegion(visitproc visit, void* arg) : visit_(visit), arg_(arg) {}

    void add_root(const TensorImpl* t) { tensors_.push_back(t); }
    void add_root(const Node* n) { nodes_.push_back(n); }
    void add_root(const ModuleBackwardHookState* s) { states_.push_back(s); }

    // Report one Python reference a member (or a FunctionCtx root) holds.
    int report(const py::object& obj) const {
        PyObject* ptr = obj.ptr();
        return ptr != nullptr ? visit_(ptr, arg_) : 0;
    }

    void offer(const TensorImplPtr& t, long copies = 0) {
        if (t && joins(t.get(), t.use_count() - copies))
            tensors_.push_back(t.get());
    }

    // Visit what every member holds, admitting objects as they qualify.
    int drain() {
        while (!tensors_.empty() || !nodes_.empty() || !states_.empty() || !slots_.empty()) {
            if (int r = drain_tensors())
                return r;
            if (int r = drain_nodes())
                return r;
            if (int r = drain_states())
                return r;
            if (int r = drain_slots())
                return r;
        }
        return 0;
    }

private:
    // Count one owning edge from a member; true the moment the last owner is
    // seen, so each object is admitted once.
    bool joins(const void* p, long owners) {
        const auto it = seen_.try_emplace(p, 0L).first;
        return ++it->second == owners;
    }

    void offer(const std::shared_ptr<Node>& n) {
        if (n && joins(n.get(), n.use_count()))
            nodes_.push_back(n.get());
    }

    void offer(const std::shared_ptr<ModuleBackwardHookState>& s) {
        if (s && joins(s.get(), s.use_count()))
            states_.push_back(s.get());
    }

    void offer(const std::shared_ptr<TensorHookSlot>& s) {
        if (s && joins(s.get(), s.use_count()))
            slots_.push_back(s.get());
    }

    int drain_tensors() {
        while (!tensors_.empty()) {
            const TensorImpl* t = tensors_.back();
            tensors_.pop_back();
            offer(t->leaf_hooks());
            offer(t->grad_as_impl(), /*copies=*/1);
            offer(t->grad_fn());
        }
        return 0;
    }

    int drain_nodes() {
        while (!nodes_.empty()) {
            const Node* n = nodes_.back();
            nodes_.pop_back();
            if (const NodeHooks* hooks = n->tensor_hooks()) {
                for (const auto& slot : hooks->slots())
                    offer(slot);
            }
            if (const auto* out_hook = dynamic_cast<const ModuleOutputHookNode*>(n)) {
                offer(out_hook->state());
            } else if (const auto* in_hook = dynamic_cast<const ModuleInputHookNode*>(n)) {
                offer(in_hook->state());
            } else if (const auto* py_node = dynamic_cast<const PythonBackwardNode*>(n)) {
                // The ctx is a Python object: one reference, and its own
                // wrapper reports what it holds.
                if (int r = report(py_node->py_ctx))
                    return r;
                if (int r = report(py_node->py_backward_fn))
                    return r;
            }
            // What the node keeps for backward: a hooked module's saved
            // output, say, that holds the module through its hooks.
            n->for_each_owned_tensor([this](const TensorImplPtr& t) { offer(t); });
            for (const auto& edge : n->next_edges())
                offer(edge.node);
        }
        return 0;
    }

    int drain_states() {
        while (!states_.empty()) {
            const ModuleBackwardHookState* s = states_.back();
            states_.pop_back();
            if (int r = report(s->pre_runner))
                return r;
            if (int r = report(s->full_runner))
                return r;
            for (const auto& g : s->grad_inputs)
                offer(g);
            for (const auto& g : s->grad_outputs)
                offer(g);
        }
        return 0;
    }

    int drain_slots() {
        while (!slots_.empty()) {
            const TensorHookSlot* s = slots_.back();
            slots_.pop_back();
            if (int r = report(s->runner))
                return r;
        }
        return 0;
    }

    visitproc visit_;
    void* arg_;
    std::unordered_map<const void*, long> seen_;
    std::vector<const TensorImpl*> tensors_;
    std::vector<const Node*> nodes_;
    std::vector<const ModuleBackwardHookState*> states_;
    std::vector<const TensorHookSlot*> slots_;
};

extern "C" int tensor_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    if (const auto* t = solely_owned<TensorImpl>(self)) {
        ExclusiveRegion region(visit, arg);
        region.add_root(t);
        if (int r = region.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int tensor_tp_clear(PyObject* self) {
    if (auto* t = solely_owned<TensorImpl>(self)) {
        // Taken out first, released after: a destructor may run Python.
        const std::shared_ptr<Node> fn = t->grad_fn();
        const std::shared_ptr<TensorImpl> gi = t->grad_as_impl();
        const std::shared_ptr<TensorHookSlot> lh = std::move(t->mutable_leaf_hooks());
        if (fn)
            t->clear_grad_fn();
        if (gi)
            t->set_grad_impl(nullptr);
    }
    return 0;
}

extern "C" int node_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    if (const auto* n = solely_owned<Node>(self)) {
        ExclusiveRegion region(visit, arg);
        region.add_root(n);
        if (int r = region.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int node_tp_clear(PyObject* self) {
    if (auto* n = solely_owned<Node>(self)) {
        n->release_tensor_hooks();
        n->release_saved();
    }
    return 0;
}

extern "C" int function_ctx_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    if (const auto* ctx = solely_owned<FunctionCtx>(self)) {
        ExclusiveRegion region(visit, arg);
        for (const auto& kv : ctx->extras()) {
            if (int r = region.report(kv.second))
                return r;
        }
        ctx->for_each_owned_tensor([&region](const TensorImplPtr& t) { region.offer(t); });
        if (int r = region.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int function_ctx_tp_clear(PyObject* self) {
    if (auto* ctx = solely_owned<FunctionCtx>(self))
        ctx->clear();
    return 0;
}

extern "C" int module_hook_state_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    if (const auto* state = solely_owned<ModuleBackwardHookState>(self)) {
        ExclusiveRegion region(visit, arg);
        region.add_root(state);
        if (int r = region.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int module_hook_state_tp_clear(PyObject* self) {
    if (auto* state = solely_owned<ModuleBackwardHookState>(self))
        state->clear();
    return 0;
}

void install_gc_hooks(PyHeapTypeObject* heap_type, traverseproc traverse, inquiry clear) {
    auto* type = &heap_type->ht_type;
    type->tp_flags |= Py_TPFLAGS_HAVE_GC;
    type->tp_traverse = traverse;
    type->tp_clear = clear;
}

}  // namespace

void release_py_object(py::object& obj) noexcept {
    PyObject* ptr = obj.release().ptr();
    if (ptr == nullptr)
        return;
    // At interpreter shutdown there is nobody to hand the reference back to.
    if (!Py_IsInitialized() || Py_IsFinalizing())
        return;
    const PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(ptr);
    PyGILState_Release(state);
}

void enable_tensor_autograd_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, tensor_tp_traverse, tensor_tp_clear);
}

void enable_node_autograd_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, node_tp_traverse, node_tp_clear);
}

void enable_function_ctx_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, function_ctx_tp_traverse, function_ctx_tp_clear);
}

void enable_module_hook_state_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, module_hook_state_tp_traverse, module_hook_state_tp_clear);
}

}  // namespace lucid
