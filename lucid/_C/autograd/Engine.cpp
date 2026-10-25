// lucid/_C/autograd/Engine.cpp
//
// Implements Engine::backward() and Engine::grad().  The algorithm is a
// standard iterative post-order DFS to compute a topological ordering of the
// backward graph, followed by a single-pass loop that executes each node's
// apply() in that order and fans the resulting input gradients out along the
// edges.  The hooks of a tensor run where its gradient is complete: on its
// producer's output slot when that node is popped, or, for a leaf, just
// before AccumulateGrad adds into ``.grad`` (TensorHooks.h).

#include "Engine.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/PhantomMode.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../ops/gfunc/Gfunc.h"
#include "AccumulateGrad.h"
#include "GraphBarrier.h"
#include "GraphOps.h"
#include "Helpers.h"
#include "Node.h"
#include "TensorHooks.h"

namespace lucid {

namespace {

constexpr const char* kSecondPass =
    "trying to backward through the graph a second time: the first backward() or "
    "autograd.grad() freed the tensors it saved — pass retain_graph=True to that call "
    "if the graph is needed again";

// Refuse a node whose saved tensors an earlier pass already freed.  Running
// it read released storage: two losses sharing a graph, each given its own
// backward() without retain_graph, took the process down with a
// segmentation fault.  An eager pass refuses a node only when what was freed
// is something its eager backward reads; a create_graph pass refuses any
// released node, as graph-mode backward reads the tensors the node was
// handed and every release frees them (Node::saved_kind).
void refuse_released(const Node& node, bool graph) {
    if (graph ? node.released() : node.saved_released())
        ErrorBuilder("backward").fail(std::string(kSecondPass) + " (" + node.node_name() + ")");
}

// One node of a severed subgraph and everything below it, by sequence
// number: the node's own and the parts of the nodes it sent gradients to.
// What a marker's subgraph held is the parts below it, shared with every
// other marker whose subgraph overlaps.  Copying the numbers into each marker
// instead made a chain of N saving nodes hold N²/2 of them, built in time to
// match: a default backward() was quadratic in the graph (LCD-542).
class SpentPart {
public:
    using Ptr = std::shared_ptr<SpentPart>;

    // The number of a part that stands for no node of its own — the union
    // a marker over several edges holds.
    static constexpr std::uint64_t kUnion = std::numeric_limits<std::uint64_t>::max();

    SpentPart(std::uint64_t sequence_nr, std::vector<Ptr> below)
        : sequence_nr_(sequence_nr),
          lo_(sequence_nr),
          hi_(sequence_nr == kUnion ? 0 : sequence_nr),
          below_(std::move(below)) {
        for (const Ptr& child : below_) {
            lo_ = std::min(lo_, child->lo_);
            hi_ = std::max(hi_, child->hi_);
        }
    }

    SpentPart(const SpentPart&) = delete;
    SpentPart& operator=(const SpentPart&) = delete;

    // Released by a loop, not by recursion: a chain of saving nodes leaves
    // a chain of parts as long as the graph, and each level of a recursive
    // release took a stack frame.
    ~SpentPart() {
        std::vector<Ptr> doomed = std::move(below_);
        while (!doomed.empty()) {
            Ptr part = std::move(doomed.back());
            doomed.pop_back();
            if (part.use_count() == 1) {
                std::move(part->below_.begin(), part->below_.end(), std::back_inserter(doomed));
                part->below_.clear();
            }
        }
    }

    std::uint64_t sequence_nr() const noexcept { return sequence_nr_; }

    std::uint64_t lo() const noexcept { return lo_; }

    std::uint64_t hi() const noexcept { return hi_; }

    const std::vector<Ptr>& below() const noexcept { return below_; }

private:
    std::uint64_t sequence_nr_;
    std::uint64_t lo_;
    std::uint64_t hi_;
    std::vector<Ptr> below_;
};

// What a node holds once a pass has freed the values its eager backward
// reads.  Clearing its edges to nothing made ``grad()`` read it as a dead end
// — reporting every input behind it unreachable, and under ``allow_unused``
// or ``backward(inputs=)`` answering ``None`` without a word — while
// replacing the root's own ``grad_fn`` hid that nothing between the root and
// an interior tensor or another branch had been freed (LCD-493).
//
// Severing at the *frontier* — the nodes whose eager-read values were
// released — keeps the save-free nodes above them wired and frees the
// subgraph below them.  Each such node's outgoing edges are replaced by a
// single edge to a :class:`SpentGraphNode` holding the parts of the subgraph
// that node could reach, so a later pass that tries to differentiate *past*
// it towards a target in that subgraph still routes into it and hits
// ``refuse_released``, while a pass that stops at an interior tensor above
// it (or follows an untouched branch) never reaches it at all.
class SpentGraphNode final : public Node {
public:
    explicit SpentGraphNode(SpentPart::Ptr below) : below_(std::move(below)) {
        mark_released(SavedKind::kEagerValues);
    }

    std::vector<Storage> apply(Storage /*grad_out*/) override {
        refuse();
        return {};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& /*grad_out*/) override {
        refuse();
        return {};
    }

    void validate_versions() override { refuse(); }

    std::string node_name() const override { return "SpentGraph"; }

    // The subgraph the node above this one was cut from, never null.  Its
    // nodes are matched by sequence number, unique for the life of the
    // process, so a node made after the subgraph was freed is never taken
    // for one of its own.
    const SpentPart::Ptr& below() const noexcept { return below_; }

private:
    static void refuse() { ErrorBuilder("backward").fail(kSecondPass); }

    SpentPart::Ptr below_;
};

// ``node`` as a spent frontier marker, or null.  Only a SpentGraphNode is
// both released and edgeless, which a walk over every node of a graph tests
// before the cast.
const SpentGraphNode* as_spent(const Node& node) {
    if (!node.next_edges().empty() || !node.saved_released())
        return nullptr;
    return dynamic_cast<const SpentGraphNode*>(&node);
}

// A gradient with a zero derivative of its own (see zero_derivative_through).
// The derivative is emitted directly rather than as ``grad * 0``, which an
// infinite upstream gradient would turn into NaN.
class ZeroDerivativeNode final : public Node {
public:
    ZeroDerivativeNode(Edge edge, Shape shape, Dtype dtype, Device device)
        : shape_(std::move(shape)), dtype_(dtype), device_(device) {
        set_next_edges({std::move(edge)});
    }

    std::vector<Storage> apply(Storage /*grad_out*/) override {
        return {make_zero_storage(shape_, dtype_, device_)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {zeros_like_op(grad_out)};
    }

    std::string node_name() const override { return "ZeroDerivativeBackward"; }

    SavedKind saved_kind() const noexcept override { return SavedKind::kNothing; }

private:
    Shape shape_;
    Dtype dtype_;
    Device device_;
};

// ── Delivery ─────────────────────────────────────────────────────────────────
//
// An ordinary node gets one pending gradient, the sum of everything routed to
// it, and runs once on it — it has one output, so that sum is its slot's
// whole gradient.  A barrier takes its gradients slot by slot instead
// (``accumulate_barrier_grad``) and ``pending`` holds only a placeholder that
// schedules it.  In graph mode only a barrier that implements GraphBarrier
// does — summing a hooked module's two outputs into one tensor was wrong — and
// any other keeps the single pending sum.
//
// A tensor a barrier produced may have hooks or ``retain_grad``, which must see
// its slot's whole gradient, and may replace it, before the barrier does.  The
// gradients of such a slot are held here instead, by (node, slot), summed as
// they arrive, and handed to the barrier through the hooks when it is popped.
// ``grad()`` holds the slots it captures the same way, so a captured gradient
// is the one the hooks returned.

template <typename Grad>
using HeldGrads = std::map<std::pair<const Node*, std::uint32_t>, Grad>;

template <typename Grad>
using SlotGrads = std::vector<std::pair<std::uint32_t, Grad>>;

using Pending = std::unordered_map<Node*, Storage>;
using GraphPending = std::unordered_map<Node*, TensorImplPtr>;

// The first arrival is copied: its producer may hand the same buffer on along
// another edge, where it is added into in place.
void hold(HeldGrads<Storage>& held, const Node* node, std::uint32_t slot, const Storage& grad) {
    auto it = held.find({node, slot});
    if (it == held.end())
        held.emplace(std::make_pair(node, slot), own_grad_copy(grad));
    else
        accumulate_into(it->second, grad);
}

void hold(HeldGrads<TensorImplPtr>& held,
          const Node* node,
          std::uint32_t slot,
          const TensorImplPtr& grad) {
    auto [it, fresh] = held.try_emplace({node, slot}, grad);
    if (!fresh)
        it->second = add_op(it->second, grad);
}

// Take what is held for ``node``'s slots, in slot order.
template <typename Grad>
SlotGrads<Grad> take_held(HeldGrads<Grad>& held, const Node* node) {
    SlotGrads<Grad> out;
    auto it = held.lower_bound({node, 0});
    while (it != held.end() && it->first.first == node) {
        out.emplace_back(it->first.second, std::move(it->second));
        it = held.erase(it);
    }
    return out;
}

template <typename Grad>
const Grad* held_value(const SlotGrads<Grad>& slots, std::uint32_t slot) {
    for (const auto& [k, grad] : slots) {
        if (k == slot)
            return &grad;
    }
    return nullptr;
}

// The gradients held for ``node``'s slots, each through the hooks of the
// tensor on that slot.
SlotGrads<Storage> fire_held(Node& node, HeldGrads<Storage>& held) {
    SlotGrads<Storage> slots = take_held(held, &node);
    for (auto& [slot, grad] : slots)
        grad = run_slot_hooks(node, slot, std::move(grad));
    return slots;
}

SlotGrads<TensorImplPtr> fire_held_for_graph(Node& node, HeldGrads<TensorImplPtr>& held) {
    SlotGrads<TensorImplPtr> slots = take_held(held, &node);
    for (auto& [slot, grad] : slots)
        grad = run_slot_hooks_for_graph(node, slot, std::move(grad));
    return slots;
}

// Route ``grad`` to slot ``slot`` of ``next`` — held when ``hold_slot`` says
// the barrier's slot has to go through hooks or be captured first.
void route(Pending& pending,
           HeldGrads<Storage>& held,
           Node* next,
           std::uint32_t slot,
           Storage grad,
           bool hold_slot) {
    if (next->is_barrier()) {
        if (hold_slot)
            hold(held, next, slot, grad);
        else
            next->accumulate_barrier_grad(slot, std::move(grad));
        pending.try_emplace(next, Storage{CpuStorage{}});
        return;
    }
    auto pit = pending.find(next);
    if (pit == pending.end()) {
        pending.emplace(next, std::move(grad));
    } else {
        if (auto* cs = std::get_if<CpuStorage>(&pit->second); cs && cs->ptr.use_count() > 1)
            pit->second = own_grad_copy(pit->second);
        accumulate_into(pit->second, grad);
    }
}

void route_for_graph(GraphPending& pending,
                     HeldGrads<TensorImplPtr>& held,
                     Node* next,
                     std::uint32_t slot,
                     const TensorImplPtr& grad,
                     bool hold_slot) {
    if (next == nullptr || !grad)
        return;
    if (hold_slot && next->is_barrier()) {
        hold(held, next, slot, grad);
        pending.try_emplace(next, nullptr);
        return;
    }
    if (auto* barrier = graph_barrier_of(next)) {
        barrier->accumulate_barrier_grad_for_graph(slot, grad);
        pending.try_emplace(next, nullptr);
        return;
    }
    auto pit = pending.find(next);
    if (pit == pending.end())
        pending.emplace(next, grad);
    else
        pit->second = pit->second ? add_op(pit->second, grad) : grad;
}

// Hand a graph-mode barrier the gradients of its held slots: slot by slot to
// a GraphBarrier, into the one pending sum otherwise.
void deliver_held_for_graph(Node& node, SlotGrads<TensorImplPtr> slots, TensorImplPtr& grad_in) {
    auto* barrier = graph_barrier_of(&node);
    for (auto& [slot, grad] : slots) {
        if (barrier != nullptr)
            barrier->accumulate_barrier_grad_for_graph(slot, std::move(grad));
        else
            grad_in = grad_in ? add_op(grad_in, grad) : std::move(grad);
    }
}

std::vector<TensorImplPtr> apply_node_for_graph(Node& node, const TensorImplPtr& grad_in) {
    if (auto* barrier = graph_barrier_of(&node))
        return barrier->apply_barrier_for_graph();
    return node.apply_for_graph(grad_in);
}

// Free what ``node`` saved for backward, now that this pass is done with it,
// and say whether that freed values its eager backward reads.  Such a node
// can never run again, so the hooks of the tensors it produced go too —
// which breaks the cycle a hook that refers to its own tensor closes through
// the node.
bool release(Node& node) {
    const Node::SavedKind freed = node.saved_kind();
    node.release_saved();
    node.mark_released(freed);
    if (node.saved_released())
        node.release_tensor_hooks();
    return freed == Node::SavedKind::kEagerValues;
}

// ── Profiler events ──────────────────────────────────────────────────────────
//
// A backward pass is one event, and each node it runs one more, named after
// the node (``backward_node_name``) and marked as backward work, so a profile
// says where a backward spends its time.  With no profiler recording, an
// event costs a thread-local read.

bool profiling() {
    const Profiler* prof = current_profiler();
    return prof != nullptr && prof->is_active();
}

class BackwardEvent {
public:
    // The whole pass, described by its first root.
    explicit BackwardEvent(const std::vector<BackwardRoot>& roots) {
        if (roots.empty() || !profiling())
            return;
        const TensorImpl& root = *roots.front().tensor;
        open("backward", root.device(), root.dtype(), root.shape());
    }

    // One node's run on an eager gradient.  An eager gradient carries no
    // shape, so the event's is its element count.  A barrier's own gradient
    // is a placeholder; the first slot it was handed stands in for it.
    BackwardEvent(const Node& node, const Storage& grad, const Storage* slot) {
        if (!profiling())
            return;
        const Storage& g = slot != nullptr ? *slot : grad;
        const Dtype dtype = storage_dtype(g);
        const auto numel = static_cast<std::int64_t>(storage_nbytes(g) / dtype_size(dtype));
        open(backward_node_name(node), storage_device(g), dtype,
             numel > 0 ? Shape{numel} : Shape{});
    }

    // One node's run under create_graph.
    BackwardEvent(const Node& node, const TensorImplPtr& grad) {
        if (!profiling())
            return;
        if (grad)
            open(backward_node_name(node), grad->device(), grad->dtype(), grad->shape());
        else
            open(backward_node_name(node), Device::CPU, Dtype::F32, Shape{});
    }

    BackwardEvent(const BackwardEvent&) = delete;
    BackwardEvent& operator=(const BackwardEvent&) = delete;

private:
    void open(std::string_view name, Device device, Dtype dtype, Shape shape) {
        scope_.emplace(name, device, dtype, std::move(shape));
        scope_->set_backward();
    }

    std::optional<OpScope> scope_;
};

// Returns true when s carries no data (zero nbytes and null pointer/array).
// Used to decide whether to synthesise a ones-valued grad_seed.
bool storage_is_empty(const Storage& s) {
    if (auto* cpu = std::get_if<CpuStorage>(&s)) {
        return cpu->nbytes == 0 && cpu->ptr == nullptr;
    }
    if (auto* gpu = std::get_if<GpuStorage>(&s)) {
        return gpu->nbytes == 0 && gpu->arr == nullptr;
    }
    return false;
}

// Compute a reverse-topological ordering of the backward graph below the
// nodes in ``roots``.
//
// Uses an iterative DFS with an explicit frame stack to avoid recursion
// overflows on deep networks.  Each frame records the node being visited and
// the index of the next edge to explore, giving the standard iterative
// post-order DFS behaviour:
//   - While the current frame still has unvisited children, push the first
//     unvisited child and advance the edge cursor.
//   - When all children are visited, append the node to `order` (post-order)
//     and pop the frame.
// Nodes already in the visited set are skipped so that shared sub-graphs are
// processed only once, whichever root they are reached from.
//
// After the DFS the vector is reversed so that every node comes before the
// nodes it sends gradients to — the roots first.
std::vector<std::shared_ptr<Node>> topo_order(const std::vector<std::shared_ptr<Node>>& roots) {
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;

    struct Frame {
        std::shared_ptr<Node> node;
        std::size_t edge_idx;
    };
    std::vector<Frame> stack;
    for (const auto& root : roots) {
        if (!visited.insert(root.get()).second)
            continue;
        stack.push_back({root, 0});
        while (!stack.empty()) {
            auto& f = stack.back();
            const auto& edges = f.node->next_edges();
            if (f.edge_idx < edges.size()) {
                const auto& edge = edges[f.edge_idx++];
                auto next = edge.node;
                if (next && visited.insert(next.get()).second) {
                    stack.push_back({std::move(next), 0});
                }
            } else {
                // All children visited: this node is ready to be appended.
                order.push_back(std::move(f.node));
                stack.pop_back();
            }
        }
    }

    // Post-order gives leaf-first ordering; reverse to get root-first.
    std::reverse(order.begin(), order.end());
    return order;
}

// The part of each node below a pass's frontier, and the frontier nodes to
// cut, with what the marker each is cut to holds (null: nothing below).
using SpentParts = std::unordered_map<const Node*, SpentPart::Ptr>;
using SpentCuts = std::vector<std::pair<Node*, SpentPart::Ptr>>;

// The part ``node`` stands for, once each node below it has one in
// ``parts``.  A node whose eager-read values were freed — by this pass or an
// earlier one — is cut (``cuts``): its marker holds the union of what is
// below it, and its own part adds only its number.  An earlier marker
// stands for what it holds already.
SpentPart::Ptr spent_part(Node& node, const SpentParts& parts, SpentCuts& cuts) {
    if (const auto* marker = as_spent(node))
        return marker->below();
    // A part linked twice is still searched once (SpentSearch), so only an
    // adjacent repeat — ``x * x`` — is dropped: a ``cat`` of every step has
    // an edge per step, and checking each against the rest would be
    // quadratic in them.
    std::vector<SpentPart::Ptr> below;
    for (const auto& edge : node.next_edges()) {
        if (!edge.node)
            continue;
        const SpentPart::Ptr& part = parts.at(edge.node.get());
        if (below.empty() || below.back() != part)
            below.push_back(part);
    }
    if (!node.saved_released())
        return std::make_shared<SpentPart>(node.sequence_nr(), std::move(below));
    if (below.empty()) {
        cuts.emplace_back(&node, nullptr);
        return std::make_shared<SpentPart>(node.sequence_nr(), std::move(below));
    }
    if (below.size() > 1)
        below = {std::make_shared<SpentPart>(SpentPart::kUnion, std::move(below))};
    cuts.emplace_back(&node, below.front());
    return std::make_shared<SpentPart>(node.sequence_nr(), std::move(below));
}

// After a pass that did not retain its graph, cut each node whose eager-read
// values were freed from the subgraph below it, replacing its outgoing edges
// with a single edge to a :class:`SpentGraphNode` holding that subgraph's
// parts, while every save-free node above them (and every untouched branch)
// stays wired.
//
// One post-order walk from ``freed``, the nodes this pass freed, gives each
// node below them its part from its children's.  Walking each freed node's
// subgraph again, one freed node after another, was what made the cut
// quadratic.  The edges are replaced only once the walk is done, so it never
// meets a node a replacement has freed.
void sever_frontier(const std::vector<std::shared_ptr<Node>>& freed) {
    if (freed.empty())
        return;
    SpentParts parts;
    parts.reserve(2 * freed.size());
    SpentCuts cuts;
    cuts.reserve(freed.size());

    struct Frame {
        Node* node;
        SpentPart::Ptr* part;
        std::size_t edge_idx;
    };
    std::vector<Frame> stack;
    for (const auto& top : freed) {
        auto [it, fresh] = parts.try_emplace(top.get());
        if (!fresh)
            continue;
        stack.push_back({top.get(), &it->second, 0});
        while (!stack.empty()) {
            Frame& f = stack.back();
            const auto& edges = f.node->next_edges();
            if (f.edge_idx < edges.size()) {
                Node* next = edges[f.edge_idx++].node.get();
                if (next == nullptr)
                    continue;
                auto [nit, unseen] = parts.try_emplace(next);
                if (unseen)
                    stack.push_back({next, &nit->second, 0});
                continue;
            }
            *f.part = spent_part(*f.node, parts, cuts);
            stack.pop_back();
        }
    }

    // In post-order a node is cut after every node below it, so whatever a
    // cut frees was cut already.
    for (auto& [node, below] : cuts) {
        std::vector<Edge> edges;
        if (below)
            edges.emplace_back(std::make_shared<SpentGraphNode>(std::move(below)), 0);
        node->set_next_edges(std::move(edges));
    }
}

// The node each root seeds — its grad_fn — or null for a leaf with none.
std::vector<std::shared_ptr<Node>> root_nodes(const std::vector<BackwardRoot>& roots) {
    std::vector<std::shared_ptr<Node>> nodes;
    for (const auto& r : roots) {
        if (r.tensor->grad_fn())
            nodes.push_back(r.tensor->grad_fn());
    }
    return nodes;
}

// Refuse a null root, and a seed that is not a gradient of its root: the
// pass reads a seed as its root's dtype, device and shape, so a short one was
// read past its end and a float64 one had its bits taken for float32.
//
// Raises
// ------
// LucidError
//     A root is null.
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     A seed is of another kind than its root.
void check_roots(const std::vector<BackwardRoot>& roots, const char* op) {
    const ErrorBuilder err(op);
    for (std::size_t i = 0; i < roots.size(); ++i) {
        const auto& r = roots[i];
        if (!r.tensor)
            err.fail("roots[" + std::to_string(i) + "] is null");
        // A phantom's graph is phantom: outside the mode its nodes would hand
        // their saved phantoms to a real backend.  Inside it the pass runs
        // phantom, and a real leaf refuses what reaches it.
        refuse_phantom_outside_mode(r.tensor->raw_storage(), std::string(op) + " of a phantom");
        if (r.seed)
            check_grad_kind(err, "the gradient of roots[" + std::to_string(i) + "]",
                            r.tensor->dtype(), r.tensor->device(), r.tensor->shape(), *r.seed);
    }
}

// ``root`` with a seed given as a bare buffer (the single-root C++ entry
// points), read as the buffer's own dtype and device, so check_roots holds it
// to the root.  A buffer carries no shape, so its size is held to the root's
// here: a short one was read past its end.  An empty buffer stands for ones.
//
// Raises
// ------
// ShapeMismatch
//     The buffer holds a number of elements other than the root's.
BackwardRoot root_with_seed(const std::shared_ptr<TensorImpl>& root, Storage seed) {
    if (storage_is_empty(seed))
        return {root, nullptr};
    const Dtype dtype = storage_dtype(seed);
    const Device device = storage_device(seed);
    const auto elements = static_cast<std::int64_t>(storage_nbytes(seed) / dtype_size(dtype));
    if (elements != static_cast<std::int64_t>(shape_numel(root->shape())))
        throw ShapeMismatch(root->shape(), {elements}, "backward: the seed of the root");
    return {root,
            std::make_shared<TensorImpl>(std::move(seed), root->shape(), dtype, device, false)};
}

// A root's seed as an eager pass carries it: ones by default, else a buffer
// of the seed's values the pass alone owns, since it adds into what it holds.
Storage eager_seed(const BackwardRoot& r) {
    const TensorImpl& t = *r.tensor;
    if (!r.seed)
        return make_ones_storage(t.shape(), t.dtype(), t.device());
    return own_grad_copy(grad_slot_buffer(ErrorBuilder("backward"), *r.seed));
}

// A root's seed under create_graph: the seed tensor itself, graph and all.
TensorImplPtr graph_seed(const BackwardRoot& r) {
    if (r.seed)
        return r.seed;
    const TensorImpl& t = *r.tensor;
    return std::make_shared<TensorImpl>(make_ones_storage(t.shape(), t.dtype(), t.device()),
                                        t.shape(), t.dtype(), t.device(), false);
}

// The seeds of the roots that are leaves with no graph, summed per leaf, so a
// leaf given twice sees its whole gradient once, through its hooks.
template <typename Grad, typename Add>
std::vector<std::pair<TensorImplPtr, Grad>>
leaf_seeds(const std::vector<BackwardRoot>& roots, std::vector<Grad>& seeds, Add add) {
    std::vector<std::pair<TensorImplPtr, Grad>> leaves;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (roots[i].tensor->grad_fn())
            continue;
        auto it = std::find_if(leaves.begin(), leaves.end(),
                               [&](const auto& l) { return l.first == roots[i].tensor; });
        if (it == leaves.end())
            leaves.emplace_back(roots[i].tensor, std::move(seeds[i]));
        else
            add(it->second, seeds[i]);
    }
    return leaves;
}

void add_eager(Storage& sum, const Storage& g) {
    accumulate_into(sum, g);
}

void add_graph(TensorImplPtr& sum, const TensorImplPtr& g) {
    sum = add_op(sum, g);
}

}  // namespace

TensorImplPtr zero_derivative_through(TensorImplPtr grad, const Edge& edge) {
    if (!grad || grad->requires_grad() || !edge.node || !GradMode::is_enabled())
        return grad;
    auto node =
        std::make_shared<ZeroDerivativeNode>(edge, grad->shape(), grad->dtype(), grad->device());
    auto alias =
        TensorImpl::make_view(grad, grad->shape(), grad->stride(), 0, /*join_family=*/false);
    alias->set_grad_fn(std::move(node));
    alias->set_grad_output_nr(0);
    alias->set_leaf(false);
    alias->set_requires_grad(true);
    return alias;
}

TensorImplPtr zero_derivative_through(TensorImplPtr grad, const TensorImplPtr& source) {
    if (!source || !source->requires_grad() || !source->grad_fn())
        return grad;
    return zero_derivative_through(std::move(grad),
                                   Edge(source->grad_fn(), source->grad_output_nr()));
}

// ── Running one node ─────────────────────────────────────────────────────────

namespace {

// Run ``node`` on its gradient once its hooks have.  The checks come after
// the hooks, not before: a hook can run a backward of its own through this
// node, which frees what it saved, or write into a tensor it saved — checked
// first, the node then read freed storage and took the process down.
std::vector<Storage> run_node(Node& node, Storage grad_in, SlotGrads<Storage> slots) {
    refuse_released(node, /*graph=*/false);
    node.validate_versions();
    const BackwardEvent event(node, grad_in, slots.empty() ? nullptr : &slots.front().second);
    if (!node.is_barrier())
        return node.apply(std::move(grad_in));
    for (auto& [slot, grad] : slots)
        node.accumulate_barrier_grad(slot, std::move(grad));
    return node.apply_barrier();
}

std::vector<TensorImplPtr>
run_node_for_graph(Node& node, TensorImplPtr grad_in, SlotGrads<TensorImplPtr> slots) {
    refuse_released(node, /*graph=*/true);
    node.validate_versions();
    // What eager backward reads by value, graph mode reads through the
    // tensors the node was handed — and an in-place write since forward
    // may have rewritten one that validate_versions does not check: a
    // saved output, or any input while the check is waived.
    node.restore_saved_for_graph();
    const BackwardEvent event(node, grad_in || slots.empty() ? grad_in : slots.front().second);
    if (node.is_barrier())
        deliver_held_for_graph(node, std::move(slots), grad_in);
    // apply_for_graph throws NotImplementedError if the op doesn't support
    // graph mode — gives the user a clear, actionable message.
    return apply_node_for_graph(node, grad_in);
}

// Validate size consistency: the number of returned gradients should equal
// the number of outgoing edges.  A mismatch is only an error when both sides
// are non-empty; an empty result with edges is permissible when the node
// intentionally produces no gradients (e.g. a stop-gradient op).
void check_result_count(const Node& node, std::size_t n) {
    const auto& edges = node.next_edges();
    if (n != edges.size() && !edges.empty() && n != 0)
        ErrorBuilder("Engine::backward").fail("input_grads/next_edges size mismatch");
}

// Hand each root's seed to its grad_fn, on the slot the root stands for.  A
// leaf with no graph was given its seed already.
template <typename Seed, typename Route>
void seed_roots(const std::vector<BackwardRoot>& roots, std::vector<Seed>& seeds, Route route_one) {
    for (std::size_t i = 0; i < roots.size(); ++i) {
        Node* const fn = roots[i].tensor->grad_fn().get();
        if (fn != nullptr)
            route_one(fn, roots[i].tensor->grad_output_nr(), std::move(seeds[i]));
    }
}

// Send an eager node's results along its edges.  A custom Function's
// ``None`` arrives as an empty storage: no gradient for that input.  Routed
// on, it became the leaf's ``.grad`` — the leaf's shape over no memory, read
// back as whatever the allocator had last left there.
template <typename Route>
void route_results(const Node& node, std::vector<Storage>& grads, Route route_one) {
    const auto& edges = node.next_edges();
    for (std::size_t i = 0; i < grads.size() && i < edges.size(); ++i) {
        Node* next = edges[i].node.get();
        if (next == nullptr || (node.empty_grad_is_none() && storage_is_empty(grads[i])))
            continue;
        route_one(next, edges[i].input_nr, std::move(grads[i]));
    }
}

// The same under create_graph, where a missing gradient is a null tensor.
template <typename Route>
void route_results(const Node& node, const std::vector<TensorImplPtr>& grads, Route route_one) {
    const auto& edges = node.next_edges();
    for (std::size_t i = 0; i < grads.size() && i < edges.size(); ++i)
        route_one(edges[i].node.get(), edges[i].input_nr, grads[i]);
}

// Pop ``node``'s pending gradient into ``grad_in``, or let a barrier that runs
// without an arriving gradient proceed with an empty placeholder.
template <typename Map, typename Grad>
bool take_pending(Map& pending, Node& node, Grad& grad_in) {
    auto it = pending.find(&node);
    if (it != pending.end()) {
        grad_in = std::move(it->second);
        pending.erase(it);
        return true;
    }
    const auto* barrier = graph_barrier_of(&node);
    return barrier != nullptr && barrier->runs_without_grad();
}

}  // namespace

// ── Graph-mode backward (create_graph=true) ───────────────────────────────────
//
// Runs the same topological traversal as backward() but uses TensorImplPtr
// throughout so that every gradient operation is itself recorded in the
// autograd graph.  Gradient accumulation at shared nodes is done via add_op
// (which creates a new backward node) rather than raw storage +=.
//
// create_graph=True implies retain_graph=True: the backward computation
// re-uses forward nodes (through apply_for_graph's saved_impl_inputs_), and a
// second backward must be able to traverse those nodes, so nothing is
// released.
static void backward_for_graph(const std::vector<BackwardRoot>& roots,
                               std::vector<TensorImplPtr> seeds) {
    // A leaf with no graph: the seed is its gradient.
    for (auto& [leaf, seed] : leaf_seeds(roots, seeds, add_graph))
        accumulate_leaf_for_graph(leaf, seed);

    const auto fns = root_nodes(roots);
    if (fns.empty())
        return;
    const auto order = topo_order(fns);

    GraphPending pending;
    HeldGrads<TensorImplPtr> held;
    const auto send = [&](Node* next, std::uint32_t slot, const TensorImplPtr& grad) {
        route_for_graph(pending, held, next, slot, grad,
                        next != nullptr && has_slot_hooks(*next, slot));
    };
    seed_roots(roots, seeds, send);

    for (const auto& node : order) {
        TensorImplPtr grad_in;
        if (!take_pending(pending, *node, grad_in))
            continue;

        // The hooks of the tensors this node produced, and retain_grad, see
        // their whole gradient before the node does.
        SlotGrads<TensorImplPtr> slots;
        if (node->is_barrier())
            slots = fire_held_for_graph(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks_for_graph(*node, 0, std::move(grad_in));

        const auto input_grads = run_node_for_graph(*node, std::move(grad_in), std::move(slots));
        check_anomaly_grads_for_graph(*node, input_grads);
        route_results(*node, input_grads, send);
    }
}

// The eager pass of Engine::backward.  ``pending`` maps each node to the
// gradient accumulated for it so far: gradients from several edges into one
// node are summed there before apply() runs.  ``held`` keeps the gradients of
// hooked barrier slots until the barrier's turn (see Delivery above).
static void eager_backward(const std::vector<BackwardRoot>& roots,
                           std::vector<Storage> seeds,
                           bool retain_graph) {
    // A leaf with no graph: the seed is its gradient — through its hooks, as
    // AccumulateGrad would run them.
    for (auto& [leaf, seed] : leaf_seeds(roots, seeds, add_eager))
        accumulate_leaf(*leaf, std::move(seed));

    const auto fns = root_nodes(roots);
    if (fns.empty())
        return;
    const auto order = topo_order(fns);

    Pending pending;
    HeldGrads<Storage> held;
    const auto send = [&](Node* next, std::uint32_t slot, Storage grad) {
        route(pending, held, next, slot, std::move(grad), has_slot_hooks(*next, slot));
    };
    seed_roots(roots, seeds, send);

    std::vector<std::shared_ptr<Node>> freed;  // released here, freeing eager-read values
    for (const auto& node : order) {
        Storage grad_in;
        if (!take_pending(pending, *node, grad_in))
            continue;  // no gradient reached it

        // The hooks of the tensors this node produced, and retain_grad, see
        // their whole gradient — and may replace it — before the node runs on
        // it.  A node with no hooks pays a null test.
        SlotGrads<Storage> slots;
        if (node->is_barrier())
            slots = fire_held(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks(*node, 0, std::move(grad_in));

        std::vector<Storage> input_grads = run_node(*node, std::move(grad_in), std::move(slots));

        // Free saved forward tensors immediately unless the caller needs the
        // graph intact for a second backward call.
        if (!retain_graph)
            if (release(*node))
                freed.push_back(node);

        check_anomaly_grads(*node, input_grads);
        check_result_count(*node, input_grads.size());
        route_results(*node, input_grads, send);
    }

    if (!retain_graph)
        sever_frontier(freed);
}

void Engine::backward(const std::shared_ptr<TensorImpl>& root,
                      Storage grad_seed,
                      bool retain_graph,
                      bool create_graph) {
    if (!root)
        ErrorBuilder("Engine::backward").fail("root is null");
    backward({root_with_seed(root, std::move(grad_seed))}, retain_graph, create_graph);
}

void Engine::backward(const std::vector<BackwardRoot>& roots,
                      bool retain_graph,
                      bool create_graph) {
    check_roots(roots, "Engine::backward");
    // Barriers discard what an earlier pass left in them (GraphBarrier.h).
    const BackwardPass pass;
    // The reference runs a backward with grad mode on only when it records
    // the graph of its gradients: a hook, a module hook or a custom
    // Function's backward then builds that graph, and otherwise builds none.
    const GradModeGuard mode(create_graph);
    const BackwardEvent event(roots);

    if (create_graph) {
        std::vector<TensorImplPtr> seeds;
        seeds.reserve(roots.size());
        for (const auto& r : roots)
            seeds.push_back(graph_seed(r));
        backward_for_graph(roots, std::move(seeds));
        return;
    }
    std::vector<Storage> seeds;
    seeds.reserve(roots.size());
    for (const auto& r : roots)
        seeds.push_back(eager_seed(r));
    eager_backward(roots, std::move(seeds), retain_graph);
}

// ── Functional gradients (no .grad written anywhere) ─────────────────────────
//
// Same traversal as backward(), with one difference that is the whole point:
// an AccumulateGrad node is never executed.  Executing it is what writes a
// leaf's .grad, so intercepting the gradient on its way in leaves every
// tensor's gradient state untouched — including leaves the caller never
// asked about.  Hooks run on the nodes the traversal visits — those on a path
// to a requested input — and retain_grad keeps what reaches them, as in
// backward().

namespace {

// Where a requested input should be captured during the traversal.
//
// Everything is keyed by node.  A leaf that has taken part in any op already
// owns an AccumulateGrad as its own ``grad_fn`` (``ensure_grad_fn`` installs
// it), so leaves and interior tensors are both reached the same way — there
// is no separate "match the leaf tensor" case to get wrong.
//
// A node can capture for several requested inputs: the outputs of one
// multi-output node share it, and so does a tensor asked for twice.  Keeping
// one index per node answered only the first of them and called the rest
// unreachable.
struct CaptureTargets {
    std::unordered_map<const Node*, std::vector<std::size_t>> by_node;
    // Only for a tensor with no grad_fn at all, which can be captured solely
    // as the root of the differentiation.
    std::unordered_map<const TensorImpl*, std::vector<std::size_t>> orphans;
};

CaptureTargets build_targets(const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
    CaptureTargets targets;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& inp = inputs[i];
        if (!inp)
            ErrorBuilder("Engine::grad").fail("inputs[" + std::to_string(i) + "] is null");
        if (inp->grad_fn())
            targets.by_node[inp->grad_fn().get()].push_back(i);
        else
            targets.orphans[inp.get()].push_back(i);
    }
    return targets;
}

// The requested inputs this node captures for, or null.
const std::vector<std::size_t>* capture_slots(const Node* node, const CaptureTargets& targets) {
    auto it = targets.by_node.find(node);
    return it == targets.by_node.end() ? nullptr : &it->second;
}

// Whether a gradient arriving at slot ``input_nr`` of ``barrier`` is one a
// requested input asked for.
//
// A barrier takes its gradients slot by slot, and ``pending`` holds only a
// placeholder for it — so the gradient of a tensor a barrier produced (a
// hooked module's output, one output of a multi-output Function) is held on
// its way in, from what reaches that tensor's own slot.  Capturing the
// placeholder handed back a tensor over no memory, and reading it crashed
// the process.
bool captures_at_barrier(const Node* barrier,
                         std::uint32_t input_nr,
                         const CaptureTargets& targets,
                         const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
    const auto* slots = capture_slots(barrier, targets);
    if (slots == nullptr)
        return false;
    return std::any_of(slots->begin(), slots->end(),
                       [&](std::size_t s) { return inputs[s]->grad_output_nr() == input_nr; });
}

// Whether a spent marker's subgraph held a requested input.
//
// A marker has no edges, so the walk cannot follow the gradient past it.
// Read as a dead end, it hid every input in the subgraph it stands for: a
// second grad() reported them unreachable — and under allow_unused, or from
// backward(inputs=), answered None without a word — where the reference
// refuses the pass.
//
// Markers whose subgraphs overlap share the parts below the overlap, and the
// answer for each part searched is kept for the next marker, so one grad()
// searches each part once.  What is kept is an answer, not a visit: a part
// is recorded as reaching no request only once everything below it has been
// searched, and every part on the way to one that does is recorded as
// reaching it.  Taking a part already visited for a dead end would answer
// for the second marker what was true only of the first.
class SpentSearch {
public:
    explicit SpentSearch(const CaptureTargets& targets) {
        wanted_.reserve(targets.by_node.size());
        for (const auto& target : targets.by_node)
            wanted_.push_back(target.first->sequence_nr());
        std::sort(wanted_.begin(), wanted_.end());
    }

    bool reaches(const SpentGraphNode& marker) {
        bool hit = enter(*marker.below());
        while (!hit && !stack_.empty()) {
            Frame& f = stack_.back();
            if (f.next == f.part->below().size()) {
                stack_.pop_back();
                continue;
            }
            const SpentPart* next = f.part->below()[f.next++].get();
            hit = enter(*next);
        }
        for (const Frame& f : stack_)
            known_[f.part] = true;
        stack_.clear();
        return hit;
    }

private:
    struct Frame {
        const SpentPart* part;
        std::size_t next;
    };

    // Whether ``part`` is known to reach a request or is one; otherwise it
    // is searched next, unless it was searched already or no request falls
    // within its descendant sequence-number range [lo, hi] (LCD-568).
    bool enter(const SpentPart& part) {
        const auto wit = std::lower_bound(wanted_.begin(), wanted_.end(), part.lo());
        if (wit == wanted_.end() || *wit > part.hi())
            return false;
        auto [it, fresh] = known_.try_emplace(&part, false);
        if (!fresh)
            return it->second;
        if (std::binary_search(wit, wanted_.end(), part.sequence_nr())) {
            it->second = true;
            return true;
        }
        stack_.push_back({&part, 0});
        return false;
    }

    std::vector<std::uint64_t> wanted_;
    std::unordered_map<const SpentPart*, bool> known_;
    std::vector<Frame> stack_;
};

// The nodes from which some requested input is still reachable.
//
// backward() has to walk the whole graph — every leaf is a destination.
// grad() does not: only the paths between the root and the requested
// inputs carry a gradient anybody asked for, so anything off them is work
// with no result to show for it.  Pruning it is not merely an
// optimisation.  Asking for the gradient with respect to an interior
// tensor is how a vector-Jacobian product is taken mid-graph — the
// divergence of a continuous flow's vector field, a gradient penalty, a
// Hessian-vector product — and continuing past that tensor drags the
// traversal into whatever produced it, which under create_graph must then
// supply a double-backward formula it was never asked to have.
//
// One reverse sweep suffices: ``order`` is topological, so a node's
// successors have already been decided by the time it is examined.
//
// A spent marker reaches what its subgraph held (SpentSearch), so it is sent
// its gradient and refuses the pass there, as backward() refuses it.
std::unordered_set<const Node*> reaching_nodes(const std::vector<std::shared_ptr<Node>>& order,
                                               const CaptureTargets& targets) {
    std::unordered_set<const Node*> reaching;
    std::optional<SpentSearch> spent;  // made at the first marker
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const Node* node = it->get();
        bool hit = targets.by_node.find(node) != targets.by_node.end();
        if (const auto* marker = hit ? nullptr : as_spent(*node)) {
            if (!spent)
                spent.emplace(targets);
            hit = spent->reaches(*marker);
        }
        if (!hit) {
            for (const auto& edge : node->next_edges()) {
                if (edge.node && reaching.find(edge.node.get()) != reaching.end()) {
                    hit = true;
                    break;
                }
            }
        }
        if (hit)
            reaching.insert(node);
    }
    return reaching;
}

// Whether this node's gradient has anywhere left to go.
//
// Distinct from membership in ``reaching``: a requested input reaches
// itself, but its gradient is captured rather than propagated unless a
// *further* request sits upstream of it.  An AccumulateGrad has no edges, so
// it never propagates — executing one is precisely what writes a leaf's
// .grad, which this path never does.  A spent marker that reaches a request
// runs, so that it refuses — and it is in ``reaching`` only then, as it is no
// requested input's grad_fn.
bool propagates(const Node* node, const std::unordered_set<const Node*>& reaching) {
    if (as_spent(*node) != nullptr)
        return reaching.find(node) != reaching.end();
    for (const auto& edge : node->next_edges()) {
        if (edge.node && reaching.find(edge.node.get()) != reaching.end())
            return true;
    }
    return false;
}

// The leaf a captured AccumulateGrad stands for, when its hooks are to run
// on what is captured for it — as AccumulateGrad would run them before
// writing ``.grad``.
TensorImplPtr hooked_leaf(const Node& node) {
    const auto* acc = dynamic_cast<const AccumulateGrad*>(&node);
    if (acc == nullptr)
        return nullptr;
    TensorImplPtr leaf = acc->leaf().lock();
    return leaf && leaf->requires_grad() ? leaf : nullptr;
}

// Where the traversal of Engine::grad goes, decided once for every root.
struct GradPlan {
    const std::vector<std::shared_ptr<TensorImpl>>& inputs;
    const CaptureTargets& targets;
    std::unordered_set<const Node*> reaching;

    // Off every path to a requested input, a node is sent nothing: none of
    // it runs in this pass, and a barrier handed a gradient it never ran on
    // kept it — a multi-output Function, which does not discard an earlier
    // pass's gradients, added them into the next backward()'s.
    bool on_path(const Node* node) const {
        return node != nullptr && reaching.find(node) != reaching.end();
    }

    // The barrier slots whose gradient is held (see Delivery): those whose
    // hooks run in this pass, those a requested input reads, and every slot
    // of a barrier that will not run — for the same reason as above.
    bool hold_at(const Node* next, std::uint32_t slot) const {
        return next->is_barrier() &&
               (has_slot_hooks(*next, slot) || captures_at_barrier(next, slot, targets, inputs) ||
                !propagates(next, reaching));
    }
};

// A root that is a leaf with no graph differentiates to itself: dy/dy is the
// seed, through the leaf's own hooks — once, on the sum of its seeds.
template <typename Grad, typename Add, typename Capture>
void capture_leaf_roots(const std::vector<BackwardRoot>& roots,
                        std::vector<Grad>& seeds,
                        const CaptureTargets& targets,
                        std::vector<TensorImplPtr>& results,
                        Add add,
                        Capture capture) {
    for (auto& [leaf, seed] : leaf_seeds(roots, seeds, add)) {
        auto it = targets.orphans.find(leaf.get());
        if (it == targets.orphans.end())
            continue;
        const TensorImplPtr value = capture(*leaf, std::move(seed));
        for (const std::size_t s : it->second)
            results[s] = value;
    }
}

// Capture what reaches ``node`` under create_graph for the inputs it stands
// for.
void capture_for_graph(const Node& node,
                       const TensorImplPtr& grad_in,
                       const SlotGrads<TensorImplPtr>& slots,
                       const GradPlan& plan,
                       std::vector<TensorImplPtr>& results) {
    const auto* wanted = capture_slots(&node, plan.targets);
    if (wanted == nullptr)
        return;
    const TensorImplPtr leaf = hooked_leaf(node);
    TensorImplPtr leaf_grad;
    for (const std::size_t s : *wanted) {
        TensorImplPtr captured = grad_in;
        if (node.is_barrier()) {
            const auto* held_grad = held_value(slots, plan.inputs[s]->grad_output_nr());
            captured = held_grad == nullptr ? nullptr : *held_grad;
        }
        if (leaf && captured) {
            // Once, however many times the leaf is asked for.
            if (!leaf_grad)
                leaf_grad = run_leaf_hooks_for_graph(*leaf, gradient_in_dtype_of(captured, leaf));
            captured = leaf_grad;
        }
        results[s] = gradient_in_dtype_of(captured, plan.inputs[s]);
    }
}

void grad_for_graph(const std::vector<BackwardRoot>& roots,
                    std::vector<TensorImplPtr> seeds,
                    const std::vector<std::shared_ptr<Node>>& order,
                    const GradPlan& plan,
                    std::vector<TensorImplPtr>& results) {
    GraphPending pending;
    HeldGrads<TensorImplPtr> held;
    const auto send = [&](Node* next, std::uint32_t slot, const TensorImplPtr& grad) {
        if (plan.on_path(next))
            route_for_graph(pending, held, next, slot, grad, plan.hold_at(next, slot));
    };
    seed_roots(roots, seeds, send);

    for (const auto& node : order) {
        TensorImplPtr grad_in;
        if (!take_pending(pending, *node, grad_in))
            continue;
        // Off every path to a requested input: nothing of it runs.
        if (!plan.on_path(node.get()))
            continue;

        const bool runs = propagates(node.get(), plan.reaching);

        // Hooks before the checks, as in backward().
        SlotGrads<TensorImplPtr> slots;
        if (node->is_barrier())
            slots = fire_held_for_graph(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks_for_graph(*node, 0, std::move(grad_in));

        capture_for_graph(*node, grad_in, slots, plan, results);
        if (!runs)
            continue;

        const auto input_grads = run_node_for_graph(*node, std::move(grad_in), std::move(slots));
        check_anomaly_grads_for_graph(*node, input_grads);
        route_results(*node, input_grads, send);
    }
}

// Capture what reaches ``node`` for the inputs it stands for.
void capture_eager(Node& node,
                   const Storage& grad_in,
                   const SlotGrads<Storage>& slots,
                   bool runs,
                   const GradPlan& plan,
                   std::vector<TensorImplPtr>& results) {
    const auto* wanted = capture_slots(&node, plan.targets);
    if (wanted == nullptr)
        return;
    const TensorImplPtr leaf = hooked_leaf(node);
    std::optional<Storage> leaf_grad;
    for (const std::size_t s : *wanted) {
        const auto& inp = plan.inputs[s];
        const Storage* captured = &grad_in;
        if (node.is_barrier())
            captured = held_value(slots, inp->grad_output_nr());
        if (captured == nullptr)
            continue;
        if (leaf) {
            // Once, however many times the leaf is asked for.
            if (!leaf_grad)
                leaf_grad = run_leaf_hooks(*leaf, *captured);
            captured = &*leaf_grad;
        }
        // A node that runs is handed the buffer too, and another target may
        // already hold it.
        const auto* cs = std::get_if<CpuStorage>(captured);
        Storage value =
            (runs || (cs && cs->ptr.use_count() > 1)) ? own_grad_copy(*captured) : *captured;
        results[s] = std::make_shared<TensorImpl>(std::move(value), inp->shape(), inp->dtype(),
                                                  inp->device(), false);
    }
}

void grad_eager(const std::vector<BackwardRoot>& roots,
                std::vector<Storage> seeds,
                const std::vector<std::shared_ptr<Node>>& order,
                const GradPlan& plan,
                bool retain_graph,
                std::vector<TensorImplPtr>& results) {
    Pending pending;
    HeldGrads<Storage> held;
    const auto send = [&](Node* next, std::uint32_t slot, Storage grad) {
        if (plan.on_path(next))
            route(pending, held, next, slot, std::move(grad), plan.hold_at(next, slot));
    };
    seed_roots(roots, seeds, send);

    std::vector<std::shared_ptr<Node>> freed;  // released here, freeing eager-read values
    for (const auto& node : order) {
        Storage grad_in;
        if (!take_pending(pending, *node, grad_in))
            continue;
        // Off every path to a requested input: nothing of it runs.
        if (!plan.on_path(node.get()))
            continue;

        const bool runs = propagates(node.get(), plan.reaching);

        // Hooks before the checks, as in backward().
        SlotGrads<Storage> slots;
        if (node->is_barrier())
            slots = fire_held(*node, held);
        else if (node->tensor_hooks() != nullptr)
            grad_in = run_slot_hooks(*node, 0, std::move(grad_in));

        capture_eager(*node, grad_in, slots, runs, plan, results);
        if (!runs)
            continue;

        std::vector<Storage> input_grads = run_node(*node, std::move(grad_in), std::move(slots));
        if (!retain_graph)
            if (release(*node))
                freed.push_back(node);
        check_anomaly_grads(*node, input_grads);
        route_results(*node, input_grads, send);
    }

    if (!retain_graph)
        sever_frontier(freed);
}

}  // namespace

std::vector<TensorImplPtr> Engine::grad(const std::shared_ptr<TensorImpl>& root,
                                        Storage grad_seed,
                                        const std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                        bool retain_graph,
                                        bool create_graph) {
    if (!root)
        ErrorBuilder("Engine::grad").fail("root is null");
    return grad({root_with_seed(root, std::move(grad_seed))}, inputs, retain_graph, create_graph);
}

std::vector<TensorImplPtr> Engine::grad(const std::vector<BackwardRoot>& roots,
                                        const std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                        bool retain_graph,
                                        bool create_graph) {
    check_roots(roots, "Engine::grad");
    // Barriers discard what an earlier pass left in them (GraphBarrier.h).
    const BackwardPass pass;
    // Grad mode as in Engine::backward.
    const GradModeGuard mode(create_graph);
    const BackwardEvent event(roots);

    const CaptureTargets targets = build_targets(inputs);
    std::vector<TensorImplPtr> results(inputs.size());

    const auto fns = root_nodes(roots);
    const auto order = topo_order(fns);
    const GradPlan plan{inputs, targets, reaching_nodes(order, targets)};

    if (create_graph) {
        std::vector<TensorImplPtr> seeds;
        seeds.reserve(roots.size());
        for (const auto& r : roots)
            seeds.push_back(graph_seed(r));
        capture_leaf_roots(roots, seeds, targets, results, add_graph,
                           [](const TensorImpl& leaf, TensorImplPtr seed) {
                               return run_leaf_hooks_for_graph(leaf, std::move(seed));
                           });
        grad_for_graph(roots, std::move(seeds), order, plan, results);
        return results;
    }

    std::vector<Storage> seeds;
    seeds.reserve(roots.size());
    for (const auto& r : roots)
        seeds.push_back(eager_seed(r));
    capture_leaf_roots(
        roots, seeds, targets, results, add_eager, [](const TensorImpl& leaf, Storage seed) {
            return std::make_shared<TensorImpl>(run_leaf_hooks(leaf, std::move(seed)), leaf.shape(),
                                                leaf.dtype(), leaf.device(), false);
        });
    grad_eager(roots, std::move(seeds), order, plan, retain_graph, results);
    return results;
}

}  // namespace lucid
