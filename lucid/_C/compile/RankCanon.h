// lucid/_C/compile/RankCanon.h
//
// Rank canonicalisation of a traced graph before it is emitted (LCD-920).
//
// MPSGraph runs a compute op that reads or writes a tensor of rank 5 or more
// as a loop over the dims above the last four, one dispatch per slice
// (~4.5 us each): ``x * w`` over ``(6144, 16, 2, 16, 2)`` takes 28 ms where the
// same elements as ``(6144, 512)`` take 2 ms.  A rank-5+ tensor that is only
// the input or output of a reshape costs nothing, and a ``reshape -> op ->
// reshape`` sandwich is folded when ops follow one another.  So the cure is
// not in any emitter: it is one pass over the :class:`TraceGraph` that wraps
// an op whose layout is free to change in views, leaving the op itself to run
// at the lowest rank its operands' broadcast pattern allows.
//
// What the pass guarantees
// ------------------------
// * Insert-only.  Every id of the original graph keeps its id, shape and
//   producer's name: ``y = f(a, b)`` becomes ``a' = reshape(a)``,
//   ``b' = reshape(b)``, ``y' = f(a', b')``, ``y = reshape(y')``.  A graph
//   output, a parameter, a loss, a custom-backward region's ids and the
//   optimizer's ghost-gradient ids therefore resolve exactly as before.
// * The backward needs no rule of its own.  The VJP walker differentiates the
//   rewritten graph, so a gradient is accumulated at ``a'`` and ``b'`` at the
//   canonical rank, with one ``reshape`` VJP back to ``a`` and ``b``.  What can
//   still add two rank-5+ gradients is ``BackwardContext::accumulate_grad``,
//   which flattens them through :func:`rank_canon_wants`.
// * A view of one id at one shape is made once, so every consumer of ``a``
//   that agrees on a layout shares ``a'``.
// * A view is never made of a view.  MPSGraph does not fold every ``reshape``
//   chain (a permute between a rank-6 view and another costs a pass over the
//   data: 1.5 -> 3.9 ms at 12.8M elements), so a view reads the root: the feed
//   behind a recorded ``reshape``, or the canonical ``y'`` a rewritten ``y`` came
//   from.  A recorded ``reshape``-family op that reads a rewritten ``y`` reads
//   ``y'`` for the same reason; ``y`` itself stays for everything else.
// * ``dynamic_batch`` never merges axis 0: the symbolic batch stays its own
//   axis, which is what ``reshape_dynamic_aware`` can follow.
//
// A rewrite is a loss when it costs a fusion MPSGraph would have made.  A
// ``conv3d`` followed by ``relu`` runs the activation inside the convolution
// (4.9 -> 6.6 ms at lead 4 when wrapped, still ~25% slower at lead 128), so the
// pass acts only above a lead threshold (:func:`rank_canon_min_lead`) and
// leaves an op that reads a convolution's result alone.
//
// Layout of the pass
// ------------------
// :class:`RankRule` is the unit a card adds — elementwise, permute, softmax and
// matmul batch dims so far, reductions next: ``handles`` names the ops it owns
// and ``plan`` returns the canonical shape of each input and of the output (and
// the axis attributes that move with it, and the further nodes of an op split in
// steps).  The driver, :class:`RankCanon`, owns everything else — id minting,
// view de-duplication, region index remapping — so a rule is pure shape
// arithmetic.
//
// A permute of a feed is left as recorded (MPSGraph transposes a feed in place at
// full speed; the rewrite made it 14-80% slower).
//
// A rewrite is also dropped when a consumer of its result outside the reshape family
// would still read it at rank 5+ (an op no rule owns, one a rule declined, a
// custom-backward region): MPSGraph then materialises the view, and the rewrite costs
// more than it saves.  A result flows on only to rewritten ops, views, reductions (which
// read a view for free) and graph outputs.
//
// Attention needs no exemption: ``MpsBuilder``'s fused-attention workaround
// (``collect_softmax_outputs``) runs on the rewritten graph and follows the views.
//
// Feeds
// -----
// A placeholder read through an in-graph ``reshape`` is materialised by MPSGraph (a
// rank-5 feed read flat costs 1.24 ms against 0.65 for the same elements fed flat), where
// the same ``reshape`` of a computed tensor is an alias.  :func:`canonical_feed_shapes`
// names the rank-5+ feeds the builder may declare at the shape their views want instead;
// the executable binds the caller's buffer at that shape (the logical one stays everywhere
// outside it, as for a rank-5+ output).
//
// Pure C++: no MPSGraph types here, so the pass is testable from Python.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../api.h"
#include "../core/fwd.h"
#include "TraceIR.h"

namespace lucid::compile {

// Ranks below this keep their emitted shape: the slowness starts at rank 5.
inline constexpr std::size_t kRankCanonMinRank = 5;

// Default lead (product of the dims above the last four) a free rewrite needs.
inline constexpr std::int64_t kRankCanonMinLead = 64;

// Default lead a rewrite that adds a pass over the data needs (a permute split into
// several permutes).  Scales with the override: ``kRankCanonCostlyFactor`` times it.
inline constexpr std::int64_t kRankCanonMinLeadCostly = 256;
inline constexpr std::int64_t kRankCanonCostlyFactor = kRankCanonMinLeadCostly / kRankCanonMinLead;

// Environment override, read at every compile.  ``0`` rewrites every rank-5+
// op (for tests), a negative value turns the pass off.
inline constexpr const char* kRankCanonMinLeadEnv = "LUCID_COMPILE_RANK_CANON_MIN_LEAD";

// The lead threshold in force now.  Negative when the pass is off.
LUCID_API std::int64_t rank_canon_min_lead();

// Product of the dims above the last four — the slice count MPSGraph loops
// over.  A symbolic (negative) dim among them makes the lead unknown, which
// reads as "large": ``INT64_MAX``.  1 for rank <= 4.
LUCID_API std::int64_t rank_lead(const std::vector<std::int64_t>& shape);

// Whether a tensor of ``shape`` is worth keeping out of MPSGraph's rank-5+
// kernels: rank >= 5, the pass is on, and ``rank_lead(shape)`` reaches the
// threshold.  The single place the policy lives; the pass and the backward's
// gradient accumulation both ask it.
LUCID_API bool rank_canon_wants(const std::vector<std::int64_t>& shape);

// Like :func:`rank_canon_wants`, against the costly threshold: a rewrite that
// costs an extra pass over the data pays for itself only at a larger lead.
LUCID_API bool rank_canon_wants_costly(const std::vector<std::int64_t>& shape);

// The result of coalescing the axes of an op's operands: the output's shape and
// each operand's, all at one lower rank.  A rank-0 operand keeps ``{}``.
struct BroadcastLayout {
    std::vector<std::int64_t> out;
    std::vector<std::vector<std::int64_t>> operands;
};

// Merge adjacent axes of ``out`` that every operand treats alike — an operand
// either spans both (``S[i] == out[i]``) or is broadcast along both
// (``S[i] == 1``) — and drop axes of size 1, which sit between any two blocks
// for free.  ``operands`` are right-aligned against ``out`` as in NumPy; each
// may have a lower rank, an empty one is a scalar.  ``keep_lead`` pins axis 0
// as a block of its own.  Nullopt when an operand does not broadcast to ``out``
// or when nothing is left to merge into (all axes of size 1).
LUCID_API std::optional<BroadcastLayout>
coalesce_broadcast(const std::vector<std::vector<std::int64_t>>& operands,
                   const std::vector<std::int64_t>& out,
                   bool keep_lead);

// Shape and dtype of an id the graph produced, or an external feed.
struct RankOperand {
    std::vector<std::int64_t> shape;
    Dtype dtype = Dtype::F32;
};

// What a rule can ask of the graph while planning one node.
class LUCID_API RankEnv {
public:
    RankEnv(const TraceGraph& graph,
            const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
            bool dynamic_batch);

    // Nullptr when the id is unknown (a rule leaves such a node alone).
    const RankOperand* find(TensorId id) const;
    // Name of the op that produced ``id``; empty for a feed.
    const std::string& producer(TensorId id) const;
    bool dynamic_batch() const { return dynamic_batch_; }
    // Largest id seen anywhere (feeds, ops, regions) — new ids start above it.
    TensorId max_id() const { return max_id_; }

private:
    std::unordered_map<TensorId, RankOperand> operands_;
    std::unordered_map<TensorId, std::string> producers_;
    bool dynamic_batch_;
    TensorId max_id_ = -1;
};

// An op's attributes, as recorded.
using RankAttrs = std::unordered_map<std::string, AttributeValue>;

// A further node of the same op, reading the previous node's result: the next step
// of a permute that does not fit in one rank <= 4 permute.
struct RankStage {
    // The shape the previous result is read at (a view is inserted when it differs).
    std::vector<std::int64_t> in_shape;
    std::vector<std::int64_t> out_shape;
    RankAttrs attrs;
};

// How one node is to be rewritten.  ``input_shapes[k]`` is the canonical shape
// to view input ``k`` at; empty leaves the input as it is (a scalar, or already
// canonical).  ``out_shape`` is the canonical shape of the node's first output and
// ``extra_outputs`` those of the others (a ``split_at`` has one per piece; each is
// viewed back under its original id).
// ``attrs`` replaces the node's attributes when an axis attribute moves with the
// layout (``dim``, ``permutation``).  ``then`` lists the nodes that follow the
// first one when the op is split; the last node's result is viewed back at the
// shape the original node produced.
struct RankRewrite {
    std::vector<std::vector<std::int64_t>> input_shapes;
    std::vector<std::int64_t> out_shape;
    std::vector<std::vector<std::int64_t>> extra_outputs;
    std::optional<RankAttrs> attrs;
    std::vector<RankStage> then;
};

// A per-op-family rule of the pass.
class LUCID_API RankRule {
public:
    virtual ~RankRule() = default;
    // Whether this rule owns ``node``'s op.
    virtual bool handles(const OpNode& node) const = 0;
    // The rewrite for ``node``, or nullopt to leave it as recorded (too small,
    // unknown operand, layout that does not coalesce below rank 5, ...).  Only
    // called for a node ``handles`` accepted.
    virtual std::optional<RankRewrite> plan(const OpNode& node, const RankEnv& env) const = 0;
};

// The rules in force, in lookup order.
LUCID_API const std::vector<std::unique_ptr<RankRule>>& rank_rules();

// The rewritten graph.  Holds the original untouched when no op needs a
// rewrite, so a trace under the threshold costs one scan.
class LUCID_API RankCanon {
public:
    RankCanon(const TraceGraph& graph,
              const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
              bool dynamic_batch);

    // The graph to emit: ``graph`` itself, or its rewrite.
    const TraceGraph& graph() const { return rewritten_ ? *rewritten_ : *original_; }
    // Views added (``reshape`` ops inserted around rewritten nodes).
    std::size_t views_added() const { return views_added_; }
    std::size_t nodes_rewritten() const { return nodes_rewritten_; }

private:
    void rewrite(const RankEnv& env,
                 const std::vector<std::optional<RankRewrite>>& plans,
                 const std::vector<bool>& inside);

    const TraceGraph* original_;
    std::unique_ptr<TraceGraph> rewritten_;
    std::size_t views_added_ = 0;
    std::size_t nodes_rewritten_ = 0;
};

// The shapes to declare feed placeholders at, by feed id: only a feed that is rank 5+
// above the lead gate and is read by nothing but pure views (``reshape`` family) of the
// rewritten graph ``graph``, and is not in ``kept`` (an id the builder reads or returns
// as it is: parameters, state, outputs).  The shape is the rank <= 4 shape most of its
// views produce, so those views become identity reshapes.  A symbolic batch is excluded
// by the caller.  Not exported: an executable records the result, nothing else reads it.
using FeedShapes = std::unordered_map<TensorId, std::vector<std::int64_t>>;
FeedShapes canonical_feed_shapes(const TraceGraph& graph,
                                 const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                                 const std::unordered_set<TensorId>& kept);

}  // namespace lucid::compile
