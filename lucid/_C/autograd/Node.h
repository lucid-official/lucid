// lucid/_C/autograd/Node.h
//
// Foundational types of Lucid's define-by-run autograd graph: ``Edge`` and
// ``Node``.
//
// Every differentiable operation produced during a forward pass instantiates
// a concrete ``Node`` subclass and links it to its inputs through a vector of
// ``Edge`` objects.  After the forward completes, :class:`Engine` walks the
// resulting DAG in reverse topological order, invoking :meth:`Node::apply` on
// each visited node to propagate gradients toward the leaf tensors.

#pragma once

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

#include <cxxabi.h>

#include "../api.h"
#include "../core/Storage.h"

namespace lucid {

class Node;
class NodeHooks;
class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

// A directed edge in the backward computation graph.
//
// An ``Edge`` connects a producer node to a specific input slot
// (``input_nr``) of a consumer node.  The owning ``shared_ptr`` keeps the
// consumer alive while at least one producer references it, forming the
// ownership chain that mirrors the forward computation.  ``input_nr`` is
// the positional index into the consumer's :meth:`Node::apply` output
// vector where the arriving gradient must land.
//
// Attributes
// ----------
// node : std::shared_ptr<Node>
//     The consumer node that should receive a gradient along this edge.
// input_nr : std::uint32_t
//     Positional input index on ``node`` that this edge targets.  The
//     engine writes the produced gradient into ``apply``'s output at
//     this slot.
//
// Notes
// -----
// An invalid edge (``node == nullptr``) marks a structural hole — for
// example an input that does not require grad — and is skipped by the
// engine without raising.
//
// See Also
// --------
// :class:`Node` — consumes edges as ``next_edges_`` during backward.
struct LUCID_API Edge {
    std::shared_ptr<Node> node;
    std::uint32_t input_nr = 0;

    // Default-constructed empty edge with ``node == nullptr``.
    Edge() = default;

    // Construct an edge pointing at consumer ``n`` slot ``i``.
    //
    // Parameters
    // ----------
    // n : std::shared_ptr<Node>
    //     Consumer node.  Ownership is moved in.
    // i : std::uint32_t, optional
    //     Input slot index on the consumer (default ``0``).
    Edge(std::shared_ptr<Node> n, std::uint32_t i = 0) : node(std::move(n)), input_nr(i) {}

    // Returns whether this edge points to a live consumer node.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when ``node != nullptr``, ``false`` for empty edges.
    bool is_valid() const { return node != nullptr; }
};

// Abstract base class for every node in the backward computation graph.
//
// A ``Node`` is the unit of work that :class:`Engine` executes during
// :func:`backward`.  Each concrete subclass saves whatever tensors and
// scalars it needs at forward time (typically into its own member fields)
// and implements :meth:`apply` to compute input gradients from the incoming
// output gradient via the chain rule.
//
// The reference framework's ``Function`` corresponds to a Python-side
// :class:`lucid.autograd.Function`; on the C++ side that role is split into
// a forward op (free function or struct method) plus this ``Node``
// subclass.  Subclassing ``Node`` directly is unusual — most engine-resident
// ops inherit through :class:`AutogradNode` / :class:`FuncOp`, which add
// fixed-size storage for saved inputs and version checking.
//
// Attributes
// ----------
// sequence_nr_ : std::uint64_t
//     Globally-unique creation order assigned via :func:`next_sequence_nr`.
//     The engine uses this to break topological ties so that nodes closer
//     to the loss are executed first.
// next_edges_ : std::vector<Edge>
//     Outgoing edges — one per input tensor of the forward op.  Populated
//     by the op builder at forward time and read-only thereafter.
// saved_versions_ : std::vector<std::int64_t>
//     Version counters captured at forward.  :class:`AutogradNode` keeps
//     one per input slot; a node that subclasses ``Node`` directly keeps
//     one per tensor it holds for backward, in an order of its own.
//     :meth:`validate_versions` compares exactly the ones whose values the
//     backward reads against the live counters, to catch in-place writes
//     that would silently corrupt gradients.
// tensor_hooks_ : std::unique_ptr<NodeHooks>
//     Hooks and ``retain_grad`` targets of the tensors this node produced,
//     by output slot; null for the common node that has none.
//
// Notes
// -----
// **Lifetime.** Nodes are jointly owned by the :class:`TensorImpl` that
// produced them (via ``grad_fn``) and by the ``Edge`` objects of consumer
// nodes.  Once :func:`backward` completes and ``grad_fn`` is cleared on the
// root, the chain of ``shared_ptr`` references collapses and all temporary
// nodes are destroyed.
//
// **Thread safety.** ``Node`` is not thread-safe.  The engine runs the
// backward graph on a single thread; concurrent :func:`backward` invocations
// on the same graph are undefined behaviour.
//
// See Also
// --------
// :class:`Edge` — the directed link between nodes.
// :class:`AutogradNode` — CRTP base with saved-input slots and version
//     checking that most ops actually inherit from.
class LUCID_API Node : public std::enable_shared_from_this<Node> {
public:
    // Construct a node and assign the next monotonically increasing
    // sequence number via :func:`next_sequence_nr`.
    Node();

    // Polymorphic destructor; subclasses are deleted through ``Node*``.
    // Out of line: the tensor hooks it owns are a complete type only in
    // Node.cpp, which keeps pybind11 out of this header.
    virtual ~Node();

    // Compute input gradients from the upstream output gradient.
    //
    // This is the core chain-rule contribution implemented by each concrete
    // backward node.  The returned vector has one entry per outgoing edge —
    // i.e. per input to the corresponding forward op — and the engine
    // dispatches entry ``i`` along ``next_edges_[i]``.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial \mathcal{L} / \partial y$ matching
    //     the forward output's shape, dtype, and device.  Implementations
    //     may assume it is non-null.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One :class:`Storage` per forward input.  An empty / default
    //     :class:`Storage` at position ``i`` signals that no gradient flows
    //     to that input.
    //
    // Math
    // ----
    // For a forward op $y = f(x_1, \ldots, x_n)$, ``apply`` returns
    //
    // $$
    // \bar{x}_i = \frac{\partial \mathcal{L}}{\partial x_i}
    //           = J_f^{(i)\,T}\, \bar{y},
    // $$
    //
    // where $J_f^{(i)}$ is the Jacobian of $f$ with respect to $x_i$.
    virtual std::vector<Storage> apply(Storage grad_out) = 0;

    // Whether this node is a barrier that batches gradients from many edges
    // before executing.  Regular ops return ``false``; barrier nodes
    // (e.g. ``AccumulateGrad``) override to ``true``.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if the engine should route incoming gradients through
    //     :meth:`accumulate_barrier_grad` / :meth:`apply_barrier` instead
    //     of :meth:`apply`.
    virtual bool is_barrier() const noexcept { return false; }

    // Whether an empty storage among this node's results means "no gradient
    // for that input" rather than a gradient of zero elements.  The nodes
    // that call into Python say ``None`` that way; a native node's empty
    // result is a real gradient of a zero-size tensor, which a CPU
    // allocation of zero bytes also leaves empty.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if the engine should drop an empty result instead of
    //     routing it on — routed, it became a leaf's ``.grad`` over no
    //     memory.
    virtual bool empty_grad_is_none() const noexcept { return false; }

    // Accumulate an arriving gradient into this barrier's internal buffer.
    //
    // Parameters
    // ----------
    // input_nr : std::uint32_t
    //     The slot the gradient is destined for.
    // grad : Storage
    //     The gradient produced by an upstream node.
    //
    // Raises
    // ------
    // std::runtime_error
    //     Always, unless overridden — calling barrier accumulation on a
    //     non-barrier node is a programmer error.
    virtual void accumulate_barrier_grad(std::uint32_t /*input_nr*/, Storage /*grad*/) {
        throw std::runtime_error("barrier accumulation called on a regular autograd node");
    }

    // Flush an accumulated barrier and produce the consolidated gradients.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One :class:`Storage` per outgoing edge (same contract as
    //     :meth:`apply`).
    //
    // Raises
    // ------
    // std::runtime_error
    //     Always, unless overridden — calling barrier flush on a non-barrier
    //     node is a programmer error.
    virtual std::vector<Storage> apply_barrier() {
        throw std::runtime_error("barrier apply called on a regular autograd node");
    }

    // Graph-recording variant of :meth:`apply` for higher-order autograd.
    //
    // Invoked by the engine when ``create_graph=True``: the backward
    // computation itself is performed through :class:`TensorImpl`-based ops
    // so that the resulting gradient tensors carry their own ``grad_fn``,
    // enabling differentiation of gradients (Hessians, double-backward).
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient as a live :class:`TensorImpl`, possibly with a
    //     ``grad_fn`` of its own.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     One :class:`TensorImpl` per forward input, each possibly
    //     carrying autograd metadata.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If the concrete node has not overridden this method —
    //     create-graph is opt-in per op.  The message names the offending
    //     op via :meth:`node_name`.
    virtual std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) {
        throw std::runtime_error(
            "create_graph=True is not yet supported for op '" + std::string(node_name()) +
            "'. "
            "Open an issue or use retain_graph=True with multiple backward calls instead.");
    }

    // Return a short human-readable identifier for this node.
    //
    // Returns
    // -------
    // std::string
    //     Defaults to ``"unknown"``; concrete nodes override to expose
    //     their ``schema_v1.name`` (e.g. ``"matmul"``, ``"linear"``).
    //
    // Notes
    // -----
    // Used in error messages emitted by :meth:`apply_for_graph` and in
    // debugger / profiler output.
    virtual std::string node_name() const {
        // The C++ type, demangled, rather than a literal "unknown".
        //
        // Only a handful of nodes overrode this, so every other one
        // reported ``op 'unknown'`` — which made "create_graph=True is
        // not yet supported for op 'unknown'" the single largest
        // unexplained failure in the audit: 115 symbols, all naming the
        // same non-op.  A diagnostic that cannot say what it is about
        // costs more than the feature it is reporting on.
        //
        // ``typeid`` is available because this class is polymorphic.
        // Nodes that override this with their schema name still win.
        //
        // The name is demangled rather than had a digit prefix stripped
        // off it.  Stripping assumed the flat Itanium form
        // ``12MulBackward`` — a length then the identifier — and that
        // shape only occurs for a class in the *global* namespace.
        // Every node here lives in ``namespace lucid``, so the real
        // mangling is ``N5lucid11MulBackwardE``: it opens with ``N``,
        // not a digit, so the loop never advanced and the whole mangled
        // string was handed back as the op name.  Nodes in the anonymous
        // namespace inside it came back worse still
        // (``N5lucid12_GLOBAL__N_113WhereBackwardE``).
        const char* mangled = typeid(*this).name();
        if (mangled == nullptr)
            return "unknown";

        int status = 0;
        char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
        const bool demangled_ok = (status == 0 && demangled != nullptr);
        std::string name(demangled_ok ? demangled : mangled);
        std::free(demangled);

        if (demangled_ok) {
            // Keep the innermost component: ``lucid::MulBackward`` and
            // ``lucid::(anonymous namespace)::WhereBackward`` both report
            // just the node's own name.  A ``::`` inside a template
            // argument belongs to the argument rather than to the
            // enclosing scope, so the search stops at the first ``<``.
            const std::size_t template_start = name.find('<');
            const std::size_t sep = name.rfind("::", template_start);
            if (sep != std::string::npos)
                name = name.substr(sep + 2);
        } else {
            // Demangling failed, so fall back to the flat form: it is
            // still right for a global-namespace node, and a mangled
            // string beats no string at all.
            std::size_t start = 0;
            while (start < name.size() && std::isdigit(static_cast<unsigned char>(name[start])))
                ++start;
            if (start < name.size())
                name = name.substr(start);
        }

        return name.empty() ? "unknown" : name;
    }

    // Refuse backward when a tensor whose values this node reads has been
    // written in place since forward.
    //
    // A node validates exactly the tensors whose values its backward reads —
    // no more, or a write into an input the formula never reads (``add``,
    // ``sum``, a view) is refused where the reference allows it; no fewer, or
    // the write reaches the gradient in silence.  The default has nothing to
    // check: it is right only for a node that holds no tensor for backward.
    // :class:`AutogradNode` guards the inputs it saved by value; a node that
    // holds a tensor any other way overrides this (``tools/check_saved_versions.py``
    // refuses a node class that holds one without saying how it is guarded).
    //
    // Raises
    // ------
    // VersionMismatch
    //     (Through overrides) when such a tensor's live version counter
    //     differs from the value snapshotted during forward.
    virtual void validate_versions() {}

    // Release every :class:`Storage` saved for backward, allowing the
    // memory to be reclaimed once the engine no longer needs this node.
    //
    // Notes
    // -----
    // Default is a no-op.  Subclasses override to reset their saved
    // slots to empty :class:`CpuStorage`.  Called automatically by the
    // engine after :meth:`apply` returns, unless ``retain_graph=True``.
    // What it frees is said by :meth:`saved_kind`.
    virtual void release_saved() {}

    // Make the tensors :meth:`apply_for_graph` reads the ones this node was
    // handed at forward time again.
    //
    // Graph-mode backward reads its inputs through the tensors it was
    // handed, where eager backward reads the Storage it saved by value — and
    // not every write between the two passes :meth:`validate_versions`.  A
    // saved *output* is not version-checked: after ``y = x.sigmoid()``, a
    // ``y.mul_(2)`` (metal; the CPU refuses that write) or a ``y.exp_()``
    // under ``no_grad`` gave ``grad(y.sum(), x, create_graph=True)`` the
    // derivative at the written ``y``, while eager backward used the saved
    // one.  Nor is anything under ``allow_mutation_on_saved_tensors``, whose
    // promise is the values from before the write.  Rebuilt, those reads
    // answer what eager backward answers.
    //
    // Called by the engine after :meth:`validate_versions` and before
    // :meth:`apply_for_graph`.  Eager backward does not call it.  The default
    // has nothing to restore.  Declared after the other virtuals so their
    // slots keep their places.
    //
    // Raises
    // ------
    // LucidError
    //     (Through overrides) when such a tensor's values were not kept in a
    //     form the forward-time tensor can be rebuilt from.
    virtual void restore_saved_for_graph() {}

    // What :meth:`release_saved` frees, asked just before it runs.
    //
    // ``kNothing``: nothing a pass frees — the node can run again in any
    // mode (``AccumulateGrad``, which a leaf keeps from graph to graph).
    // ``kGraphInputs``: only the tensors :meth:`apply_for_graph` reads, so an
    // eager pass may run the node again and a create_graph pass may not.
    // ``kEagerValues``: values :meth:`apply` reads, so no pass may.  The
    // default, for a node that does not say: a node holding a tensor in a
    // member of its own reads it at backward, and refusing a second pass is
    // what keeps it from reading freed memory.
    enum class SavedKind : std::uint8_t { kNothing, kGraphInputs, kEagerValues };
    virtual SavedKind saved_kind() const noexcept { return SavedKind::kEagerValues; }

    // Call ``fn`` on every tensor this node holds a strong reference to — what
    // it keeps for backward — so that the cyclic garbage collector, which
    // reaches the graph only through the engine, can see past the node.  The
    // default holds none.
    virtual void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const {
        static_cast<void>(fn);
    }

    // Monotonically-increasing identifier assigned at construction.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Creation order.  The engine compares sequence numbers to break
    //     topological ties, executing later-created nodes first so that
    //     gradients flow from loss toward leaves.
    std::uint64_t sequence_nr() const { return sequence_nr_; }

    // Overwrite the sequence number.  Used by serialisation / replay tools
    // that need to reconstruct a graph deterministically.
    //
    // Parameters
    // ----------
    // n : std::uint64_t
    //     New sequence number to install.
    void set_sequence_nr(std::uint64_t n) { sequence_nr_ = n; }

    // Read-only view of this node's outgoing edges.
    //
    // Returns
    // -------
    // const std::vector<Edge>&
    //     Edge list whose ``i``-th entry is the destination of the gradient
    //     :meth:`apply` returns at position ``i``.
    const std::vector<Edge>& next_edges() const { return next_edges_; }

    // Install the outgoing edges produced by the forward op builder.
    //
    // Parameters
    // ----------
    // edges : std::vector<Edge>
    //     One edge per forward input.  Moved into the node.
    //
    // Notes
    // -----
    // Called exactly once at forward time; reading via :meth:`next_edges`
    // is the only legal access during backward.
    void set_next_edges(std::vector<Edge> edges) { next_edges_ = std::move(edges); }

    // Read-only view of the input version counters captured at forward.
    //
    // Returns
    // -------
    // const std::vector<std::int64_t>&
    //     One counter per forward input, in input order.
    const std::vector<std::int64_t>& saved_versions() const { return saved_versions_; }

    // Install the per-input version counters captured at forward time.
    //
    // Parameters
    // ----------
    // v : std::vector<std::int64_t>
    //     One counter per forward input.  Moved into the node.
    void set_saved_versions(std::vector<std::int64_t> v) { saved_versions_ = std::move(v); }

    // Record that a pass released this node, freeing what ``freed`` says —
    // :meth:`saved_kind`, read before :meth:`release_saved` ran.  The
    // version counters go too: nothing they guard is left to check.
    void mark_released(SavedKind freed) noexcept {
        if (freed == SavedKind::kNothing)
            return;
        release_state_ |= kReleased;
        if (freed == SavedKind::kEagerValues)
            release_state_ |= kFreedEagerValues;
        std::vector<std::int64_t>().swap(saved_versions_);
    }

    // Whether a pass released what this node held for backward.  A
    // create_graph pass refuses such a node: graph-mode backward reads the
    // tensors the node was handed, and a release frees them all.
    bool released() const noexcept { return (release_state_ & kReleased) != 0; }

    // Whether a release freed values this node's eager backward reads, so
    // that every pass refuses it.  A node that saved none of them — ``add``,
    // ``sum``, a view — runs eagerly again, as the reference allows.
    bool saved_released() const noexcept { return (release_state_ & kFreedEagerValues) != 0; }

    // The hooks and ``retain_grad`` targets of the tensors this node
    // produced, by output slot (TensorHooks.h), or null when there are none
    // — the only cost a node without hooks pays.
    NodeHooks* tensor_hooks() const noexcept { return tensor_hooks_.get(); }

    // The hook slots, made empty on first use.
    NodeHooks& ensure_tensor_hooks();

    // Drop every hook slot.  Called by the engine once :meth:`release_saved`
    // has made this node impossible to run again: a hook's closure that
    // refers to the tensor it is on closes a cycle — tensor, node, hook,
    // tensor — that Python's collector cannot see through the engine.
    void release_tensor_hooks() noexcept;

    // Advance any saved-output version baseline recorded on this node to ``v``
    // when an out-of-place op's result is rebound in-place onto a destination
    // tensor whose version counter moves to ``v``.
    virtual void advance_saved_output_version(std::int64_t /*v*/) noexcept {}

protected:
    // Refuse when ``live`` has moved past the version recorded at ``k`` in
    // :attr:`saved_versions_`.  ``input_idx`` names the tensor in the error
    // as the op's argument position.  Nothing is checked when no version was
    // recorded at ``k`` or the tensor is gone — nothing can then reach the
    // gradient through it.
    //
    // Raises
    // ------
    // VersionMismatch
    //     The tensor was written in place since forward and the check is not
    //     waived (``allow_mutation_on_saved_tensors``).
    void check_saved_version(const std::weak_ptr<TensorImpl>& live,
                             std::size_t k,
                             std::string_view op_name,
                             std::size_t input_idx) const;

    std::uint64_t sequence_nr_;
    std::vector<Edge> next_edges_;
    std::vector<std::int64_t> saved_versions_;

private:
    static constexpr std::uint8_t kReleased = 1;
    static constexpr std::uint8_t kFreedEagerValues = 2;

    std::unique_ptr<NodeHooks> tensor_hooks_;
    // What releases have freed (:meth:`mark_released`).
    std::uint8_t release_state_ = 0;
};

// Return the next globally-unique node sequence number and advance the
// shared counter.
//
// Returns
// -------
// std::uint64_t
//     A value strictly greater than any previously returned.
//
// Notes
// -----
// Thread-safe via relaxed atomic increment.  Relaxed ordering is sufficient
// because the engine only requires that any two nodes' sequence numbers be
// distinct and totally ordered — not that the ordering match the global
// memory order across threads.
LUCID_API std::uint64_t next_sequence_nr();

}  // namespace lucid
