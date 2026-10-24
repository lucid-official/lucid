// lucid/_C/autograd/Node.cpp
//
// Implements the global sequence-number counter, the Node constructor and
// destructor, and the node's tensor-hook slots.  The counter is a single
// process-wide atomic so that every node created in any thread receives a
// unique, monotonically increasing ID.

#include "Node.h"

#include <new>
#include <utility>

#include "Helpers.h"
#include "TensorHooks.h"

namespace lucid {

namespace {
// Process-wide counter.  Relaxed ordering is sufficient because the only
// guarantee needed is uniqueness, not synchronisation with other memory.
std::atomic<std::uint64_t> g_seq{0};

using PendingNodes = std::vector<std::shared_ptr<Node>>;

// The nodes the release running on this thread has still to let go of, or
// null when none is running.  A pointer rather than an object: it has no
// destructor, so a node let go of while thread-local storage is being torn
// down still reads a valid value.
thread_local PendingNodes* t_pending = nullptr;

// Move onto ``pending`` the nodes ``edges`` hold the last reference to.  An
// edge whose node something else still holds lets go of it here, which
// cannot destroy it.
void defer_release(std::vector<Edge>& edges, PendingNodes& pending) noexcept {
    for (Edge& edge : edges) {
        if (edge.node.use_count() != 1) {
            edge.node.reset();
            continue;
        }
        try {
            pending.push_back(std::move(edge.node));
        } catch (const std::bad_alloc&) {
            // No memory to grow the worklist: release the node here.  That
            // recurses one level, which beats terminating in a destructor.
            edge.node.reset();
        }
    }
}
}  // namespace

std::uint64_t next_sequence_nr() {
    return g_seq.fetch_add(1, std::memory_order_relaxed);
}

// Assign a sequence number at construction so that every node can be sorted
// deterministically by the engine even before next_edges_ is populated.
Node::Node() : sequence_nr_(next_sequence_nr()) {}

// The graph below a node is let go of by a loop, not by recursion.  Each
// edge owns the node it points at, so dropping the last tensor of a chain of
// N nodes destroyed N nodes nested one inside the next, a few stack frames
// each: about 200,000 of them overflowed the main thread's stack (LCD-564),
// and far fewer a worker thread's.  Now the outermost destructor on a thread
// keeps a worklist, and every node destroyed while it runs — through an edge,
// or through a tensor a node held whose grad_fn it was — hands its own edges
// to that list instead of releasing them where it stands.  The members of a
// derived node are gone by the time this body runs, so a tensor one of them
// held has already dropped its grad_fn and the use counts read here are
// final.  Python objects a node holds are released by its own destructor, as
// before, under the GIL that destructor takes.
Node::~Node() {
    if (next_edges_.empty())
        return;
    if (PendingNodes* running = t_pending) {
        defer_release(next_edges_, *running);
        return;
    }
    PendingNodes pending;
    t_pending = &pending;
    defer_release(next_edges_, pending);
    while (!pending.empty()) {
        std::shared_ptr<Node> node = std::move(pending.back());
        pending.pop_back();
        node.reset();
    }
    t_pending = nullptr;
}

NodeHooks& Node::ensure_tensor_hooks() {
    if (!tensor_hooks_)
        tensor_hooks_ = std::make_unique<NodeHooks>();
    return *tensor_hooks_;
}

void Node::release_tensor_hooks() noexcept {
    tensor_hooks_.reset();
}

void Node::check_saved_version(const std::weak_ptr<TensorImpl>& live,
                               std::size_t k,
                               std::string_view op_name,
                               std::size_t input_idx) const {
    if (k < saved_versions_.size())
        check_version_match(live, saved_versions_[k], op_name, input_idx);
}

}  // namespace lucid
