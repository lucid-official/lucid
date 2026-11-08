// lucid/_C/compile/RankCanon.cpp
//
// See RankCanon.h for the contract; this file is the policy, the broadcast
// coalescer, the elementwise rule and the driver that applies a plan.

#include "RankCanon.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <map>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "../core/Dtype.h"
#include "../core/TensorImpl.h"

namespace lucid::compile {

// ────────────────────────────────────────────────────────────────────
// Policy
// ────────────────────────────────────────────────────────────────────

std::int64_t rank_canon_min_lead() {
    const char* raw = std::getenv(kRankCanonMinLeadEnv);
    if (raw == nullptr || *raw == '\0')
        return kRankCanonMinLead;
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0')
        return kRankCanonMinLead;
    return v < 0 ? -1 : static_cast<std::int64_t>(v);
}

std::int64_t rank_lead(const std::vector<std::int64_t>& shape) {
    if (shape.size() <= 4)
        return 1;
    std::int64_t lead = 1;
    const std::size_t above = shape.size() - 4;
    for (std::size_t i = 0; i < above; ++i) {
        if (shape[i] < 0)
            return std::numeric_limits<std::int64_t>::max();
        if (shape[i] != 0 && lead > std::numeric_limits<std::int64_t>::max() / shape[i])
            return std::numeric_limits<std::int64_t>::max();
        lead *= shape[i];
    }
    return lead;
}

bool rank_canon_wants(const std::vector<std::int64_t>& shape) {
    if (shape.size() < kRankCanonMinRank)
        return false;
    const std::int64_t threshold = rank_canon_min_lead();
    return threshold >= 0 && rank_lead(shape) >= threshold;
}

bool rank_canon_wants_costly(const std::vector<std::int64_t>& shape) {
    if (shape.size() < kRankCanonMinRank)
        return false;
    const std::int64_t threshold = rank_canon_min_lead();
    if (threshold < 0)
        return false;
    const std::int64_t costly =
        threshold > std::numeric_limits<std::int64_t>::max() / kRankCanonCostlyFactor
            ? threshold
            : threshold * kRankCanonCostlyFactor;
    return rank_lead(shape) >= costly;
}

// ────────────────────────────────────────────────────────────────────
// Broadcast coalescing
// ────────────────────────────────────────────────────────────────────

namespace {

using Dims = std::vector<std::int64_t>;

// One run of merged axes: its size in the output, in each operand, and which
// operands span it (rather than being broadcast along it).
struct Block {
    std::int64_t out = 1;
    Dims operands;
    std::vector<bool> spans;
    bool pinned = false;
};

// ``operands`` right-aligned against ``out`` with 1s on the left; a scalar stays
// empty.  Nullopt when an operand does not broadcast to ``out``.
std::optional<std::vector<Dims>> align_operands(const std::vector<Dims>& operands,
                                                const Dims& out) {
    std::vector<Dims> padded(operands.size());
    for (std::size_t k = 0; k < operands.size(); ++k) {
        if (operands[k].empty())
            continue;
        if (operands[k].size() > out.size())
            return std::nullopt;
        padded[k].assign(out.size() - operands[k].size(), 1);
        padded[k].insert(padded[k].end(), operands[k].begin(), operands[k].end());
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (padded[k][i] != 1 && padded[k][i] != out[i])
                return std::nullopt;
        }
    }
    return padded;
}

Block axis_block(const std::vector<Dims>& padded, const Dims& out, std::size_t axis, bool pinned) {
    Block block;
    block.out = out[axis];
    block.pinned = pinned;
    block.operands.assign(padded.size(), 1);
    block.spans.assign(padded.size(), false);
    for (std::size_t k = 0; k < padded.size(); ++k) {
        if (padded[k].empty())
            continue;
        block.operands[k] = padded[k][axis];
        block.spans[k] = padded[k][axis] != 1;
    }
    return block;
}

// An axis joins the block before it when every operand treats the two alike.
bool joins(const Block& tail, const Block& axis) {
    return !tail.pinned && !axis.pinned && tail.spans == axis.spans;
}

void absorb(Block& tail, const Block& axis) {
    tail.out *= axis.out;
    for (std::size_t k = 0; k < tail.operands.size(); ++k)
        tail.operands[k] *= axis.operands[k];
}

}  // namespace

std::optional<BroadcastLayout>
coalesce_broadcast(const std::vector<Dims>& operands, const Dims& out, bool keep_lead) {
    const auto padded = align_operands(operands, out);
    if (!padded)
        return std::nullopt;

    std::vector<Block> blocks;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const bool pinned = keep_lead && i == 0;
        if (out[i] == 1 && !pinned)
            continue;
        Block axis = axis_block(*padded, out, i, pinned);
        if (!blocks.empty() && joins(blocks.back(), axis))
            absorb(blocks.back(), axis);
        else
            blocks.push_back(std::move(axis));
    }
    if (blocks.empty())
        return std::nullopt;

    BroadcastLayout layout;
    layout.operands.assign(operands.size(), {});
    for (const Block& b : blocks) {
        layout.out.push_back(b.out);
        for (std::size_t k = 0; k < operands.size(); ++k) {
            if (!(*padded)[k].empty())
                layout.operands[k].push_back(b.operands[k]);
        }
    }
    return layout;
}

// ────────────────────────────────────────────────────────────────────
// RankEnv
// ────────────────────────────────────────────────────────────────────

RankEnv::RankEnv(const TraceGraph& graph,
                 const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                 bool dynamic_batch)
    : dynamic_batch_(dynamic_batch) {
    auto note = [&](TensorId id) { max_id_ = std::max<TensorId>(max_id_, id); };
    for (const auto& [id, impl] : external_feeds) {
        note(id);
        if (impl)
            operands_[id] = RankOperand{impl->shape(), impl->dtype()};
    }
    auto note_node = [&](const OpNode& node) {
        for (TensorId id : node.inputs)
            note(id);
        for (const TensorMeta& m : node.outputs)
            note(m.id);
    };
    for (const OpNode& node : graph.ops) {
        note_node(node);
        for (const TensorMeta& m : node.outputs) {
            operands_[m.id] = RankOperand{m.shape, m.dtype};
            producers_[m.id] = node.name;
        }
    }
    for (const VjpRegion& r : graph.vjp_regions) {
        for (TensorId id : r.input_ids)
            note(id);
        for (TensorId id : r.cotangent_ids)
            note(id);
        for (TensorId id : r.input_grad_ids)
            note(id);
        for (const TensorMeta& m : r.output_metas)
            note(m.id);
        for (const OpNode& node : r.bwd_ops)
            note_node(node);
    }
    note_attention_weights(graph);
}

namespace {

using IdSet = std::unordered_set<TensorId>;
using NodeList = std::vector<const OpNode*>;

bool reads_any(const IdSet& ids, const std::vector<TensorId>& inputs) {
    return std::any_of(inputs.begin(), inputs.end(),
                       [&](TensorId id) { return id >= 0 && ids.count(id) != 0; });
}

bool makes_any(const IdSet& ids, const std::vector<TensorMeta>& outputs) {
    return std::any_of(outputs.begin(), outputs.end(),
                       [&](const TensorMeta& m) { return ids.count(m.id) != 0; });
}

// Softmax outputs and what flows from them through a ``dropout``, in trace order.
IdSet attention_weights_of(const NodeList& nodes) {
    IdSet weights;
    for (const OpNode* n : nodes) {
        const bool makes =
            n->name == "softmax" || (n->name == "dropout" && reads_any(weights, n->inputs));
        for (const TensorMeta& m : n->outputs) {
            if (makes)
                weights.insert(m.id);
        }
    }
    return weights;
}

// The weights a matmul reads, directly or through the dropouts between.
IdSet weights_read_by_matmuls(const NodeList& nodes, const IdSet& weights) {
    IdSet read;
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        const OpNode& n = **it;
        const bool reads =
            n.name == "matmul" || (n.name == "dropout" && makes_any(read, n.outputs));
        for (TensorId id : n.inputs) {
            if (reads && id >= 0 && weights.count(id) != 0)
                read.insert(id);
        }
    }
    return read;
}

}  // namespace

void RankEnv::note_attention_weights(const TraceGraph& graph) {
    NodeList nodes;
    for (const OpNode& node : graph.ops)
        nodes.push_back(&node);
    for (const VjpRegion& r : graph.vjp_regions) {
        for (const OpNode& node : r.bwd_ops)
            nodes.push_back(&node);
    }
    attention_weights_ = attention_weights_of(nodes);
    matmul_weights_ = weights_read_by_matmuls(nodes, attention_weights_);
}

const RankOperand* RankEnv::find(TensorId id) const {
    const auto it = operands_.find(id);
    return it == operands_.end() ? nullptr : &it->second;
}

const std::string& RankEnv::producer(TensorId id) const {
    static const std::string none;
    const auto it = producers_.find(id);
    return it == producers_.end() ? none : it->second;
}

// ────────────────────────────────────────────────────────────────────
// ElementwiseRule
// ────────────────────────────────────────────────────────────────────

namespace {

// Ops whose result at one position depends only on the operands at that
// position (after NumPy broadcasting) and on attributes that name no axis.  Each
// emitter in OpEmitters/elementwise (and ``where``) reads nothing else from its
// tensors, so the same op over a coalesced layout computes the same values.
const std::unordered_set<std::string_view>& elementwise_ops() {
    static const std::unordered_set<std::string_view> names{
        // activations
        "relu", "sigmoid", "tanh", "silu", "gelu", "gelu_exact", "elu", "leaky_relu", "selu",
        "mish", "softplus", "hard_sigmoid", "hard_swish", "relu6",
        // unary math
        "neg", "abs", "exp", "log", "sqrt", "square", "reciprocal", "rsqrt", "arccos", "arcsin",
        "arctan", "ceil", "floor", "trunc", "round", "cube", "cube_root", "log2", "sign", "erf",
        "erfinv", "clip", "sin", "cos", "tan", "sinh", "cosh", "astype", "isfinite", "isnan",
        "isinf", "invert", "pow_scalar", "rpow_scalar",
        // binary
        "add", "sub", "mul", "div", "pow", "maximum", "minimum", "floordiv", "remainder", "fmod",
        "nextafter", "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
        "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift",
        // ternary
        "where"};
    return names;
}

// Producers whose result MPSGraph fuses an elementwise consumer into (the
// convolution's activation epilogue).  Wrapping that consumer in views ends the
// fusion, and the loss is bigger than the slice loop it would avoid.
bool fuses_into_epilogue(const std::string& producer) {
    return producer == "conv3d" || producer == "conv_transpose3d";
}

// Whether ``node`` reads a result that fuses its consumer.
bool reads_fused_producer(const OpNode& node, const RankEnv& env) {
    return std::any_of(node.inputs.begin(), node.inputs.end(), [&](TensorId id) {
        return id >= 0 && fuses_into_epilogue(env.producer(id));
    });
}

// The shapes ``node``'s inputs are read at, or nullopt when the node must stay as
// recorded: an input is unknown, complex or a convolution's result, or — under a
// symbolic batch, axis 0 of an input of the output's own rank — one is padded by
// the broadcast and has no axis to keep in place.
std::optional<std::vector<Dims>> operand_shapes(const OpNode& node, const RankEnv& env) {
    if (reads_fused_producer(node, env))
        return std::nullopt;
    const std::size_t out_rank = node.outputs[0].shape.size();
    std::vector<Dims> shapes;
    shapes.reserve(node.inputs.size());
    for (TensorId id : node.inputs) {
        const RankOperand* operand = id >= 0 ? env.find(id) : nullptr;
        if (operand == nullptr || is_complex(operand->dtype))
            return std::nullopt;
        if (env.dynamic_batch() && !operand->shape.empty() && operand->shape.size() != out_rank)
            return std::nullopt;
        shapes.push_back(operand->shape);
    }
    return shapes;
}

class ElementwiseRule final : public RankRule {
public:
    bool handles(const OpNode& node) const override {
        return elementwise_ops().count(node.name) != 0;
    }

    std::optional<RankRewrite> plan(const OpNode& node, const RankEnv& env) const override {
        if (node.outputs.size() != 1 || node.inputs.empty())
            return std::nullopt;
        const TensorMeta& out = node.outputs[0];
        if (!rank_canon_wants(out.shape) || is_complex(out.dtype))
            return std::nullopt;
        const auto shapes = operand_shapes(node, env);
        if (!shapes)
            return std::nullopt;

        const auto layout = coalesce_broadcast(*shapes, out.shape, env.dynamic_batch());
        if (!layout || layout->out.size() > 4 || layout->out.size() >= out.shape.size())
            return std::nullopt;

        RankRewrite rewrite;
        rewrite.out_shape = layout->out;
        rewrite.input_shapes.resize(shapes->size());
        for (std::size_t k = 0; k < shapes->size(); ++k) {
            if (!layout->operands[k].empty() && layout->operands[k] != (*shapes)[k])
                rewrite.input_shapes[k] = layout->operands[k];
        }
        return rewrite;
    }
};

// ────────────────────────────────────────────────────────────────────
// PermuteRule
// ────────────────────────────────────────────────────────────────────

using Perm = std::vector<std::int64_t>;
using Order = std::vector<int>;

// A permute at the lowest rank its axes allow: ``out[t] == in[perm[t]]``.
struct PermuteLayout {
    Dims in;
    Dims out;
    Perm perm;
};

bool is_permutation_of(const Perm& perm, std::size_t n) {
    if (perm.size() != n)
        return false;
    std::vector<bool> seen(n, false);
    for (std::int64_t p : perm) {
        if (p < 0 || static_cast<std::size_t>(p) >= n || seen[static_cast<std::size_t>(p)])
            return false;
        seen[static_cast<std::size_t>(p)] = true;
    }
    return true;
}

// The axes a permute has to keep: those of size 1 carry nothing (``pin_first`` keeps
// axis 0 whatever its size).
std::vector<bool> kept_axes(const Dims& shape, bool pin_first) {
    std::vector<bool> kept(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
        kept[i] = shape[i] != 1 || (pin_first && i == 0);
    return kept;
}

// Where each kept axis lands among the kept axes of the output.
std::vector<std::size_t> output_ranks(const Perm& perm, const std::vector<bool>& kept) {
    std::vector<std::size_t> rank(perm.size(), 0);
    std::size_t seen = 0;
    for (std::int64_t p : perm) {
        if (kept[static_cast<std::size_t>(p)])
            rank[static_cast<std::size_t>(p)] = seen++;
    }
    return rank;
}

// The kept input axes grouped into runs that stay adjacent, in order, in the output:
// each run's size, and the run each axis belongs to.
struct Runs {
    Dims sizes;
    std::vector<std::size_t> run_of;
};

Runs merge_runs(const Dims& shape,
                const std::vector<bool>& kept,
                const std::vector<std::size_t>& rank_out,
                bool pin_first) {
    Runs runs;
    runs.run_of.assign(shape.size(), 0);
    std::optional<std::size_t> last;
    for (std::size_t a = 0; a < shape.size(); ++a) {
        if (!kept[a])
            continue;
        const bool joins = last && !(pin_first && *last == 0) && rank_out[a] == rank_out[*last] + 1;
        if (joins)
            runs.sizes.back() *= shape[a];
        else
            runs.sizes.push_back(shape[a]);
        runs.run_of[a] = runs.sizes.size() - 1;
        last = a;
    }
    return runs;
}

// Merge the input axes of ``shape`` that stay adjacent, in order, in the output
// of ``perm`` (a block moved whole is one axis) and drop the axes of size 1.
// Nullopt when ``perm`` is not a permutation of the axes or only axes of size 1
// are left.
std::optional<PermuteLayout> coalesce_permute(const Dims& shape, const Perm& perm, bool pin_first) {
    if (!is_permutation_of(perm, shape.size()))
        return std::nullopt;
    const std::vector<bool> kept = kept_axes(shape, pin_first);
    if (std::none_of(kept.begin(), kept.end(), [](bool k) { return k; }))
        return std::nullopt;
    const Runs runs = merge_runs(shape, kept, output_ranks(perm, kept), pin_first);

    PermuteLayout layout;
    layout.in = runs.sizes;
    for (std::int64_t p : perm) {
        const auto axis = static_cast<std::size_t>(p);
        const auto run = static_cast<std::int64_t>(runs.run_of[axis]);
        if (kept[axis] && (layout.perm.empty() || layout.perm.back() != run))
            layout.perm.push_back(run);
    }
    for (std::int64_t run : layout.perm)
        layout.out.push_back(layout.in[static_cast<std::size_t>(run)]);
    return layout;
}

constexpr std::size_t kMaxSplitAxes = 6;
constexpr std::size_t kMaxSplitSteps = 3;

// ``order`` with the block ``[from, to)`` moved so that it starts at ``at`` among the
// axes that remain.
Order move_block(const Order& order, std::size_t from, std::size_t to, std::size_t at) {
    const auto first = order.begin() + static_cast<std::ptrdiff_t>(from);
    const auto last = order.begin() + static_cast<std::ptrdiff_t>(to);
    const Order block(first, last);
    Order rest(order.begin(), first);
    rest.insert(rest.end(), last, order.end());
    rest.insert(rest.begin() + static_cast<std::ptrdiff_t>(at), block.begin(), block.end());
    return rest;
}

// The permute that takes the axes of ``sizes`` from the order ``cur`` to ``next``,
// at the lowest rank it allows; nullopt when that is still above 4.
std::optional<PermuteLayout>
step_layout(const Dims& sizes, const Order& cur, const Order& next, bool pin_first) {
    Dims shape(cur.size());
    Perm perm(cur.size());
    for (std::size_t i = 0; i < cur.size(); ++i)
        shape[i] = sizes[static_cast<std::size_t>(cur[i])];
    for (std::size_t t = 0; t < cur.size(); ++t)
        perm[t] = std::find(cur.begin(), cur.end(), next[t]) - cur.begin();
    auto layout = coalesce_permute(shape, perm, pin_first);
    if (!layout || layout->perm.size() > 4)
        return std::nullopt;
    return layout;
}

// The orders one block move from ``cur`` whose step is a permute of rank <= 4.  Moving
// a block ``X`` past ``Y`` (``P X Y S`` to ``P Y X S``) is always rank 4 unless the
// pinned batch axis costs one more.
std::vector<Order> block_moves(const Dims& sizes, const Order& cur, bool pin_first) {
    const std::size_t r = cur.size();
    const std::size_t first = pin_first ? 1 : 0;
    std::vector<Order> moves;
    for (std::size_t from = first; from < r; ++from) {
        for (std::size_t to = from + 1; to <= r; ++to) {
            for (std::size_t at = first; at <= r - (to - from); ++at) {
                Order moved = move_block(cur, from, to, at);
                if (at != from && step_layout(sizes, cur, moved, pin_first))
                    moves.push_back(std::move(moved));
            }
        }
    }
    return moves;
}

// The fewest permutes of rank <= 4 whose composition is ``target`` over axes of
// ``sizes``, found by breadth-first search over the orders of the axes.  Empty when
// ``target`` is out of reach in a few steps.
std::vector<PermuteLayout> split_permute(const Dims& sizes, const Order& target, bool pin_first) {
    if (sizes.size() > kMaxSplitAxes)
        return {};
    Order start(sizes.size());
    for (std::size_t i = 0; i < start.size(); ++i)
        start[i] = static_cast<int>(i);

    std::map<Order, Order> parent{{start, start}};
    std::vector<Order> frontier{start};
    for (std::size_t depth = 0; depth < kMaxSplitSteps; ++depth) {
        std::vector<Order> next_frontier;
        for (const Order& cur : frontier) {
            for (Order& moved : block_moves(sizes, cur, pin_first)) {
                if (!parent.emplace(moved, cur).second)
                    continue;
                if (moved == target) {
                    std::vector<Order> path{target};
                    while (path.back() != start)
                        path.push_back(parent.at(path.back()));
                    std::reverse(path.begin(), path.end());
                    std::vector<PermuteLayout> steps;
                    for (std::size_t k = 0; k + 1 < path.size(); ++k)
                        steps.push_back(*step_layout(sizes, path[k], path[k + 1], pin_first));
                    return steps;
                }
                next_frontier.push_back(std::move(moved));
            }
        }
        frontier = std::move(next_frontier);
    }
    return {};
}

RankAttrs with_attr(const RankAttrs& attrs, const char* key, AttributeValue value) {
    RankAttrs copy = attrs;
    copy[key] = std::move(value);
    return copy;
}

// The one operand of a single-input node that a rule may rewrite: known, not complex,
// and not the result of a producer that fuses its consumer.  Nullptr otherwise.
const RankOperand* sole_operand(const OpNode& node, const RankEnv& env) {
    if (node.outputs.size() != 1 || node.inputs.size() != 1 || node.inputs[0] < 0)
        return nullptr;
    const RankOperand* operand = env.find(node.inputs[0]);
    if (operand == nullptr || is_complex(operand->dtype) || is_complex(node.outputs[0].dtype) ||
        reads_fused_producer(node, env))
        return nullptr;
    return operand;
}

// ``node``'s ``permutation`` attribute when it is a permutation of ``rank`` axes.
const Perm* permutation_attr(const OpNode& node, std::size_t rank) {
    const auto attr = node.attrs.find("permutation");
    const auto* perm = attr == node.attrs.end() ? nullptr : std::get_if<Perm>(&attr->second);
    return perm != nullptr && !perm->empty() && perm->size() == rank ? perm : nullptr;
}

// The permutes that stand for ``merged``: itself when it is rank <= 4, else the block
// moves that reach it — a pass over the data each, so only at the costly lead.
std::vector<PermuteLayout>
permute_steps(const PermuteLayout& merged, const TensorMeta& out, bool pin_first) {
    if (merged.perm.size() <= 4)
        return {merged};
    if (!rank_canon_wants_costly(out.shape))
        return {};
    const Order target(merged.perm.begin(), merged.perm.end());
    return split_permute(merged.in, target, pin_first);
}

RankRewrite permute_rewrite(const OpNode& node, const std::vector<PermuteLayout>& steps) {
    RankRewrite rewrite;
    rewrite.input_shapes = {steps[0].in};
    rewrite.out_shape = steps[0].out;
    rewrite.attrs = with_attr(node.attrs, "permutation", steps[0].perm);
    for (std::size_t k = 1; k < steps.size(); ++k) {
        rewrite.then.push_back(RankStage{steps[k].in, steps[k].out,
                                         with_attr(node.attrs, "permutation", steps[k].perm)});
    }
    return rewrite;
}

// ``permute`` (the op every transpose, ``mT``, ``swapaxes`` and ``movedim`` records)
// at rank <= 4: a window or patch permute usually merges to one permute, and one
// that does not is split into rank <= 4 block moves.
//
// A feed is transposed in place at full speed whatever its rank (measured: a rank-6
// patchify permute of a feed 3.1 ms, 5.6 ms after the rewrite), so only a permute of
// a computed tensor — the usual ``reshape -> permute`` — is rewritten.
class PermuteRule final : public RankRule {
public:
    bool handles(const OpNode& node) const override { return node.name == "permute"; }

    std::optional<RankRewrite> plan(const OpNode& node, const RankEnv& env) const override {
        const RankOperand* operand = sole_operand(node, env);
        if (operand == nullptr || env.producer(node.inputs[0]).empty() ||
            !rank_canon_wants(node.outputs[0].shape))
            return std::nullopt;
        const Perm* perm = permutation_attr(node, operand->shape.size());
        // A symbolic batch has to stay the leading axis for its views to follow it.
        const bool pin = env.dynamic_batch();
        if (perm == nullptr || (pin && (*perm)[0] != 0))
            return std::nullopt;
        const auto merged = coalesce_permute(operand->shape, *perm, pin);
        if (!merged)
            return std::nullopt;
        const auto steps = permute_steps(*merged, node.outputs[0], pin);
        if (steps.empty())
            return std::nullopt;
        return permute_rewrite(node, steps);
    }
};

// ────────────────────────────────────────────────────────────────────
// SoftmaxRule
// ────────────────────────────────────────────────────────────────────

// A softmax's rows: the axes before ``dim`` merged, ``dim`` itself, the axes after it
// merged; ``dim`` is where it landed.
struct SoftmaxLayout {
    Dims shape;
    std::int64_t dim = 0;
};

SoftmaxLayout softmax_layout(const Dims& shape, std::size_t axis, bool pin_batch) {
    SoftmaxLayout layout;
    // Merge ``[begin, end)`` into one axis; ``keep`` keeps it when it is of size 1.
    auto merge = [&](std::size_t begin, std::size_t end, bool keep) {
        std::int64_t rows = 1;
        for (std::size_t i = begin; i < end; ++i)
            rows *= shape[i];
        if (rows != 1 || keep)
            layout.shape.push_back(rows);
    };
    if (pin_batch) {
        merge(0, 1, true);
        merge(1, axis, false);
    } else {
        merge(0, axis, false);
    }
    layout.dim = static_cast<std::int64_t>(layout.shape.size());
    layout.shape.push_back(shape[axis]);
    merge(axis + 1, shape.size(), false);
    return layout;
}

// The ``dim`` attribute when it names an axis of a rank-``rank`` tensor.
const std::int64_t* softmax_dim(const OpNode& node, std::size_t rank) {
    const auto attr = node.attrs.find("dim");
    const auto* dim = attr == node.attrs.end() ? nullptr : std::get_if<std::int64_t>(&attr->second);
    return dim != nullptr && *dim >= 0 && static_cast<std::size_t>(*dim) < rank ? dim : nullptr;
}

// ``softmax`` / ``log_softmax`` over axis ``dim``: the axes before it merge into one
// and so do the axes after it, so the op runs at rank <= 4 on the same rows.
//
// Attention weights a matmul reads stay as recorded: the matmul's workaround finds
// them by id (see RankEnv::attention_weight).
class SoftmaxRule final : public RankRule {
public:
    bool handles(const OpNode& node) const override {
        return node.name == "softmax" || node.name == "log_softmax";
    }

    std::optional<RankRewrite> plan(const OpNode& node, const RankEnv& env) const override {
        const RankOperand* operand = sole_operand(node, env);
        if (operand == nullptr)
            return std::nullopt;
        const TensorMeta& out = node.outputs[0];
        const std::int64_t* dim = softmax_dim(node, out.shape.size());
        if (dim == nullptr || operand->shape != out.shape || !rank_canon_wants(out.shape) ||
            env.feeds_attention_matmul(out.id))
            return std::nullopt;
        const auto axis = static_cast<std::size_t>(*dim);
        const SoftmaxLayout layout =
            softmax_layout(out.shape, axis, env.dynamic_batch() && axis > 0);
        if (layout.shape.size() >= out.shape.size())
            return std::nullopt;

        RankRewrite rewrite;
        rewrite.input_shapes = {layout.shape};
        rewrite.out_shape = layout.shape;
        rewrite.attrs = with_attr(node.attrs, "dim", layout.dim);
        return rewrite;
    }
};

// ────────────────────────────────────────────────────────────────────
// MatmulRule
// ────────────────────────────────────────────────────────────────────

// The shapes of a matmul's two operands, or nullopt when the rule must leave it: an
// operand is unknown, complex, a vector, attention weights (the workaround needs the
// operand as the softmax made it) or, under a symbolic batch, not of the output's
// rank (axis 0 has to stay in place).
std::optional<std::vector<Dims>>
matmul_operands(const OpNode& node, const TensorMeta& out, const RankEnv& env) {
    std::vector<Dims> shapes;
    for (TensorId id : node.inputs) {
        const RankOperand* operand = id >= 0 ? env.find(id) : nullptr;
        if (operand == nullptr || is_complex(operand->dtype) || operand->shape.size() < 2 ||
            env.attention_weight(id))
            return std::nullopt;
        if (env.dynamic_batch() && operand->shape.size() != out.shape.size())
            return std::nullopt;
        shapes.push_back(operand->shape);
    }
    return shapes;
}

// ``batch`` followed by the matrix axes (the last two) of ``full``.
Dims with_matrix(Dims batch, const Dims& full) {
    batch.insert(batch.end(), full.end() - 2, full.end());
    return batch;
}

// ``matmul`` over batch axes: adjacent batch axes that both operands treat alike
// (spanned by both, or broadcast in the same one) merge, so the batch is at most two
// axes and each operand stays rank <= 4.  The matrices' own two axes are untouched.
class MatmulRule final : public RankRule {
public:
    bool handles(const OpNode& node) const override { return node.name == "matmul"; }

    std::optional<RankRewrite> plan(const OpNode& node, const RankEnv& env) const override {
        if (node.outputs.size() != 1 || node.inputs.size() != 2)
            return std::nullopt;
        const TensorMeta& out = node.outputs[0];
        if (is_complex(out.dtype) || !rank_canon_wants(out.shape) ||
            reads_fused_producer(node, env))
            return std::nullopt;
        const auto shapes = matmul_operands(node, out, env);
        if (!shapes)
            return std::nullopt;

        std::vector<Dims> batches;
        for (const Dims& s : *shapes)
            batches.emplace_back(s.begin(), s.end() - 2);
        const Dims out_batch(out.shape.begin(), out.shape.end() - 2);
        const auto layout = coalesce_broadcast(batches, out_batch, env.dynamic_batch());
        if (!layout || layout->out.size() > 2 || layout->out.size() + 2 >= out.shape.size())
            return std::nullopt;

        RankRewrite rewrite;
        rewrite.input_shapes.resize(2);
        for (std::size_t k = 0; k < 2; ++k) {
            Dims canonical = with_matrix(layout->operands[k], (*shapes)[k]);
            if (!layout->operands[k].empty() && canonical != (*shapes)[k])
                rewrite.input_shapes[k] = std::move(canonical);
        }
        rewrite.out_shape = with_matrix(layout->out, out.shape);
        return rewrite;
    }
};

}  // namespace

const std::vector<std::unique_ptr<RankRule>>& rank_rules() {
    static const std::vector<std::unique_ptr<RankRule>> rules = [] {
        std::vector<std::unique_ptr<RankRule>> v;
        v.push_back(std::make_unique<ElementwiseRule>());
        v.push_back(std::make_unique<PermuteRule>());
        v.push_back(std::make_unique<SoftmaxRule>());
        v.push_back(std::make_unique<MatmulRule>());
        return v;
    }();
    return rules;
}

// ────────────────────────────────────────────────────────────────────
// RankCanon
// ────────────────────────────────────────────────────────────────────

namespace {

// The ops a custom-backward region owns: [fwd_begin, fwd_end) of any region.
// The walker skips their per-op VJP and the region's own backward reads their
// ids, so they stay as recorded.
std::vector<bool> ops_inside_regions(const TraceGraph& graph) {
    std::vector<bool> inside(graph.ops.size(), false);
    for (const VjpRegion& r : graph.vjp_regions) {
        const std::size_t end = std::min(r.fwd_end, graph.ops.size());
        for (std::size_t i = r.fwd_begin; i < end; ++i)
            inside[i] = true;
    }
    return inside;
}

OpNode reshape_node(TensorId from, const TensorMeta& to) {
    OpNode node;
    node.name = "reshape";
    node.inputs = {from};
    node.outputs = {to};
    return node;
}

// Ops whose result is their input's elements in the same order at the shape the
// node records (``ReshapeFamilyEmitter``): reading one's result is reading its input.
bool is_pure_view(const OpNode& node) {
    static const std::unordered_set<std::string_view> names{"reshape", "view", "flatten", "squeeze",
                                                            "unsqueeze"};
    return names.count(node.name) != 0 && node.inputs.size() == 1 && node.outputs.size() == 1;
}

// The views the pass inserts.  MPSGraph does not fold every ``reshape`` chain (a
// permute behind a rank-6 view and in front of another costs an extra pass over the
// data), so a view is never made of a view or of a result the pass has just
// reshaped: it reads the root — the feed, or the canonical result — instead.
class ViewBook {
public:
    ViewBook(const RankEnv& env, TensorId next) : env_(env), next_(next) {}

    // A fresh id for a tensor of ``like``'s dtype and device at ``shape``.
    TensorMeta mint(const TensorMeta& like, const Dims& shape) {
        TensorMeta meta{next_++, shape, like.dtype, like.device};
        made_[meta.id] = meta;
        return meta;
    }
    TensorId next_id() const { return next_; }
    std::size_t added() const { return added_; }

    // ``node`` is a pure view of its input.
    void note_view_op(const OpNode& node) { roots_[node.outputs[0].id] = node.inputs[0]; }
    // ``canonical`` holds the elements of the original result ``original``.
    void note_canonical(TensorId original, TensorId canonical) { aliases_[original] = canonical; }

    // The canonical result standing in for ``id``, or ``id``.
    TensorId canonical_of(TensorId id) const {
        const auto it = aliases_.find(id);
        return it == aliases_.end() ? id : it->second;
    }

    // An id that holds ``id``'s elements at ``shape``; the reshape that makes it is
    // appended to ``ops`` when none exists yet.  ``like`` gives the device.
    TensorId
    view(std::vector<OpNode>& ops, TensorId id, const Dims& shape, const TensorMeta& like) {
        const TensorId source = root_of(id);
        const TensorMeta* have = meta_of(source);
        if (have != nullptr && have->shape == shape)
            return source;
        const auto key = std::make_pair(static_cast<std::int64_t>(source), shape);
        auto it = views_.find(key);
        if (it == views_.end()) {
            TensorMeta from = like;
            if (have != nullptr)
                from.dtype = have->dtype;
            it = views_.emplace(key, reshape(ops, source, mint(from, shape))).first;
        }
        return it->second;
    }

    // A reshape of ``from`` into ``to`` (a fresh id from :func:`mint`), appended to
    // ``ops``.
    TensorId reshape(std::vector<OpNode>& ops, TensorId from, const TensorMeta& to) {
        ops.push_back(reshape_node(from, to));
        ++added_;
        return to.id;
    }

private:
    TensorId root_of(TensorId id) const {
        for (;;) {
            if (const auto a = aliases_.find(id); a != aliases_.end()) {
                id = a->second;
            } else if (const auto r = roots_.find(id); r != roots_.end()) {
                id = r->second;
            } else {
                return id;
            }
        }
    }

    // Shape and dtype of ``id`` (the device is the graph's), nullptr when unknown.
    const TensorMeta* meta_of(TensorId id) {
        if (const auto it = made_.find(id); it != made_.end())
            return &it->second;
        const RankOperand* operand = env_.find(id);
        if (operand == nullptr)
            return nullptr;
        return &(made_[id] = TensorMeta{id, operand->shape, operand->dtype, Device::CPU});
    }

    const RankEnv& env_;
    TensorId next_;
    std::size_t added_ = 0;
    std::unordered_map<TensorId, TensorMeta> made_;
    std::unordered_map<TensorId, TensorId> aliases_;
    std::unordered_map<TensorId, TensorId> roots_;
    std::map<std::pair<std::int64_t, Dims>, TensorId> views_;
};

// Append ``node`` as ``plan`` rewrites it — views of its inputs, the canonical node(s),
// and a view of the last result at the shape ``node`` produced, under the original id.
void emit_rewritten(const OpNode& node, const RankRewrite& plan, ViewBook& book, TraceGraph& out) {
    const TensorMeta& result = node.outputs[0];
    OpNode canonical = node;
    for (std::size_t k = 0; k < node.inputs.size(); ++k) {
        if (!plan.input_shapes[k].empty())
            canonical.inputs[k] = book.view(out.ops, node.inputs[k], plan.input_shapes[k], result);
    }
    if (plan.attrs)
        canonical.attrs = *plan.attrs;
    TensorMeta last = book.mint(result, plan.out_shape);
    canonical.outputs[0] = last;
    out.ops.push_back(std::move(canonical));

    // A split op: each further node reads the previous result, viewed at the shape it
    // wants when that differs.
    for (const RankStage& stage : plan.then) {
        OpNode follow = node;
        follow.attrs = stage.attrs;
        TensorId read = last.id;
        if (last.shape != stage.in_shape)
            read = book.reshape(out.ops, last.id, book.mint(result, stage.in_shape));
        last = book.mint(result, stage.out_shape);
        follow.inputs = {read};
        follow.outputs[0] = last;
        out.ops.push_back(std::move(follow));
    }
    book.reshape(out.ops, last.id, result);
    book.note_canonical(result.id, last.id);
}

}  // namespace

RankCanon::RankCanon(const TraceGraph& graph,
                     const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                     bool dynamic_batch)
    : original_(&graph) {
    if (rank_canon_min_lead() < 0)
        return;
    const bool any_wide = std::any_of(graph.ops.begin(), graph.ops.end(), [](const OpNode& n) {
        return n.outputs.size() == 1 && n.outputs[0].shape.size() >= kRankCanonMinRank;
    });
    if (!any_wide)
        return;

    const RankEnv env(graph, external_feeds, dynamic_batch);
    const std::vector<bool> inside = ops_inside_regions(graph);
    std::vector<std::optional<RankRewrite>> plans(graph.ops.size());
    bool any_plan = false;
    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        if (inside[i])
            continue;
        for (const auto& rule : rank_rules()) {
            if (!rule->handles(graph.ops[i]))
                continue;
            plans[i] = rule->plan(graph.ops[i], env);
            any_plan = any_plan || plans[i].has_value();
            break;
        }
    }
    if (any_plan)
        rewrite(env, plans, inside);
}

void RankCanon::rewrite(const RankEnv& env,
                        const std::vector<std::optional<RankRewrite>>& plans,
                        const std::vector<bool>& inside) {
    const TraceGraph& src = *original_;
    auto out = std::make_unique<TraceGraph>();
    out->unsupported = src.unsupported;
    out->vjp_regions = src.vjp_regions;
    out->ops.reserve(src.ops.size() + src.ops.size() / 4);
    ViewBook book(env, std::max<TensorId>(src.next_id, env.max_id() + 1));
    std::vector<std::size_t> position(src.ops.size() + 1, 0);

    for (std::size_t i = 0; i < src.ops.size(); ++i) {
        position[i] = out->ops.size();
        const OpNode& node = src.ops[i];
        if (plans[i]) {
            emit_rewritten(node, *plans[i], book, *out);
            ++nodes_rewritten_;
            continue;
        }
        out->ops.push_back(node);
        if (inside[i] || !is_pure_view(node))
            continue;
        // A view of a rewritten result reads the canonical one.
        out->ops.back().inputs[0] = book.canonical_of(node.inputs[0]);
        book.note_view_op(node);
    }
    position[src.ops.size()] = out->ops.size();

    for (VjpRegion& region : out->vjp_regions) {
        region.fwd_begin = position[std::min(region.fwd_begin, src.ops.size())];
        region.fwd_end = position[std::min(region.fwd_end, src.ops.size())];
    }
    out->next_id = book.next_id();
    views_added_ = book.added();
    rewritten_ = std::move(out);
}

}  // namespace lucid::compile
