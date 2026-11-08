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

// The shapes ``node``'s inputs are read at, or nullopt when the node must stay as
// recorded: an input is unknown, complex or a convolution's result, or — under a
// symbolic batch, axis 0 of an input of the output's own rank — one is padded by
// the broadcast and has no axis to keep in place.
std::optional<std::vector<Dims>> operand_shapes(const OpNode& node, const RankEnv& env) {
    const std::size_t out_rank = node.outputs[0].shape.size();
    std::vector<Dims> shapes;
    shapes.reserve(node.inputs.size());
    for (TensorId id : node.inputs) {
        const RankOperand* operand = id >= 0 ? env.find(id) : nullptr;
        if (operand == nullptr || is_complex(operand->dtype) ||
            fuses_into_epilogue(env.producer(id)))
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

}  // namespace

const std::vector<std::unique_ptr<RankRule>>& rank_rules() {
    static const std::vector<std::unique_ptr<RankRule>> rules = [] {
        std::vector<std::unique_ptr<RankRule>> v;
        v.push_back(std::make_unique<ElementwiseRule>());
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
        rewrite(env, plans);
}

void RankCanon::rewrite(const RankEnv& env, const std::vector<std::optional<RankRewrite>>& plans) {
    const TraceGraph& src = *original_;
    auto out = std::make_unique<TraceGraph>();
    out->unsupported = src.unsupported;
    out->vjp_regions = src.vjp_regions;
    out->ops.reserve(src.ops.size() + src.ops.size() / 4);
    TensorId next = std::max<TensorId>(src.next_id, env.max_id() + 1);

    // One view of an id at a shape, however many ops read it that way.
    std::map<std::pair<std::int64_t, Dims>, TensorId> views;
    std::vector<std::size_t> position(src.ops.size() + 1, 0);

    for (std::size_t i = 0; i < src.ops.size(); ++i) {
        position[i] = out->ops.size();
        const OpNode& node = src.ops[i];
        if (!plans[i]) {
            out->ops.push_back(node);
            continue;
        }
        const RankRewrite& plan = *plans[i];
        const TensorMeta& result = node.outputs[0];

        OpNode canonical = node;
        for (std::size_t k = 0; k < node.inputs.size(); ++k) {
            const Dims& shape = plan.input_shapes[k];
            if (shape.empty())
                continue;
            const auto key = std::make_pair(static_cast<std::int64_t>(node.inputs[k]), shape);
            auto it = views.find(key);
            if (it == views.end()) {
                const RankOperand* operand = env.find(node.inputs[k]);
                const TensorId view_id = next++;
                out->ops.push_back(reshape_node(
                    node.inputs[k], TensorMeta{view_id, shape, operand->dtype, result.device}));
                ++views_added_;
                it = views.emplace(key, view_id).first;
            }
            canonical.inputs[k] = it->second;
        }

        const TensorId canonical_id = next++;
        canonical.outputs[0] =
            TensorMeta{canonical_id, plan.out_shape, result.dtype, result.device};
        out->ops.push_back(std::move(canonical));
        out->ops.push_back(reshape_node(canonical_id, result));
        ++views_added_;
        ++nodes_rewritten_;
    }
    position[src.ops.size()] = out->ops.size();

    for (VjpRegion& region : out->vjp_regions) {
        region.fwd_begin = position[std::min(region.fwd_begin, src.ops.size())];
        region.fwd_end = position[std::min(region.fwd_end, src.ops.size())];
    }
    out->next_id = next;
    rewritten_ = std::move(out);
}

}  // namespace lucid::compile
