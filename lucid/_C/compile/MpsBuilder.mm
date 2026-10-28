// lucid/_C/compile/MpsBuilder.mm
//
// Walks a :class:`TraceGraph` and produces a compiled MPSGraph
// executable.  See :file:`MpsBuilder.h` for the contract.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "../backend/gpu/mps/MpsBridge.h"
#include "../core/Determinism.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "MpsBuilder.h"
#include "MpsDtype.h"
#include "OpEmitters/OpEmitter.h"
#include "OpEmitters/_SignedZero.h"
#include "VjpEmitters/VjpEmitter.h"

namespace lucid::compile {

// Defined in CompiledExecutable.mm — friend-private state of the
// :class:`CompiledExecutable` aggregate we build below.
class CompiledExecutable;

namespace {

// Emit ``node`` and hold every output it bound at the dtype the trace
// declared (``held_at``).  Every op of every compile mode goes through here,
// so an emitter whose MPSGraph primitive picks its own result type cannot
// hand the next op a value of another width.  Returns the reason the build
// fails, or an empty string.
std::string emit_node(OpEmitter& emitter, BuilderContext& ctx, const OpNode& node) {
    if (!emitter.emit(ctx, node))
        return "emitter for op '" + node.name + "' returned false (unsupported variant)";
    MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
    for (const auto& meta : node.outputs) {
        auto* bound = (__bridge MPSGraphTensor*)ctx.resolve(meta.id);
        if (bound == nil)
            continue;  // a dead piece of a multi-output op, never bound
        MPSGraphTensor* held = held_at(g, bound, meta.dtype);
        if (held == nil)
            return "op '" + node.name + "' produced a value that cannot be held at its declared " +
                   "dtype " + std::string(dtype_name(meta.dtype));
        if (held != bound)
            ctx.bind(meta.id, (__bridge void*)held);
    }
    propagate_zero_splat(ctx, node);
    return {};
}

// Ops whose result is fixed by their arguments rather than by any
// tensor, so binding the trace-time buffer is the right answer.
//
// This list exists to keep a much more dangerous case out.  An op that
// never records its trace I/O leaves its node with no inputs and an
// unclaimed output; the builder then sees a node nothing consumes and
// drops it, while the *consumer* meets the result as a fresh external
// feed bound once, at trace time.  The compiled model reports success
// and answers every later input with the first one's result.  Eight ops
// were doing exactly that until 2026-09-04.
//
// A genuinely discarded op is distinguishable and does not belong here:
// it recorded its inputs, and its output is dead because nothing reads
// it — so nothing can be frozen.  Only the zero-input case is
// ambiguous, and that is what this answers.
//
// Adding a name here asserts that the op's output cannot depend on a
// tensor.  Getting that wrong reintroduces the silent freeze, so the
// default for an unknown name is to refuse the build (eager fallback),
// which is the recoverable direction.

// MPSGraph aborts — it does not decline — on a constant or a placeholder
// with a zero-length axis ("shape[0] (0) should be a strictly positive
// value for constant operation").  A compiled Fast R-CNN step died that
// way on an image with no proposals.  An empty tensor is nothing to
// accelerate: name the first one so the trace is left to eager.
static std::string first_empty_tensor(const TraceGraph& graph,
                                      const std::unordered_map<TensorId, TensorImplPtr>& feeds) {
    for (const auto& op : graph.ops)
        for (const auto& out : op.outputs)
            for (const auto d : out.shape)
                if (d == 0)
                    return "an output of op '" + op.name + "'";
    for (const auto& r : graph.vjp_regions)
        for (const auto& op : r.bwd_ops)
            for (const auto& out : op.outputs)
                for (const auto d : out.shape)
                    if (d == 0)
                        return "an output of op '" + op.name + "'";
    for (const auto& [tid, impl] : feeds)
        if (impl)
            for (const auto d : impl->shape())
                if (d == 0)
                    return "an input";
    return {};
}

static void step_loss_ancestors_region(const VjpRegion& r, std::unordered_set<TensorId>& needed) {
    bool feeds_loss = false;
    for (const auto& om : r.output_metas) {
        if (needed.count(om.id) != 0) {
            feeds_loss = true;
            break;
        }
    }
    if (!feeds_loss)
        return;
    for (std::size_t k = 0; k < r.input_ids.size() && k < r.input_grad_ids.size(); ++k) {
        if (r.input_ids[k] >= 0 && r.input_grad_ids[k] >= 0)
            needed.insert(r.input_ids[k]);
    }
}

// Every trace id ``loss_id`` depends on, itself included — one reverse walk
// over the recorded ops.  MPSGraph's ``gradientForPrimaryTensor:withTensors:``
// aborts the process ("Not a predecessor of primaryTensor") when one of the
// requested tensors does not feed the primary tensor, so no caller may ask it
// for a parameter outside this set.
static std::unordered_set<TensorId> loss_ancestors(const TraceGraph& graph, TensorId loss_id) {
    std::unordered_map<std::size_t, const VjpRegion*> region_by_end;
    for (const auto& r : graph.vjp_regions) {
        if (r.fwd_end > r.fwd_begin)
            region_by_end[r.fwd_end] = &r;
    }
    std::unordered_set<TensorId> needed{loss_id};
    std::size_t idx = graph.ops.size();
    while (idx > 0) {
        auto rit = region_by_end.find(idx);
        if (rit != region_by_end.end()) {
            step_loss_ancestors_region(*rit->second, needed);
            idx = rit->second->fwd_begin;
            continue;
        }
        const auto& op = graph.ops[--idx];
        for (const auto& out : op.outputs) {
            if (needed.count(out.id) != 0) {
                for (TensorId iid : op.inputs)
                    if (iid >= 0)
                        needed.insert(iid);
                break;
            }
        }
    }
    return needed;
}

// The first of ``param_ids`` the loss does not depend on, or -1.
static TensorId first_param_off_loss_path(const TraceGraph& graph,
                                          TensorId loss_id,
                                          const std::vector<TensorId>& param_ids) {
    const auto ancestors = loss_ancestors(graph, loss_id);
    for (TensorId pid : param_ids)
        if (ancestors.count(pid) == 0)
            return pid;
    return -1;
}

inline bool result_is_a_host_constant(std::string_view name) {
    return name == "arange" || name == "empty" || name == "eye" || name == "full" ||
           name == "linspace" || name == "logspace" || name == "ones" || name == "zeros" ||
           name == "einsum" || name == "meshgrid" || name == "tri" || name == "tril" ||
           name == "triu";
}

// The op names MPSGraph's automatic differentiation was measured to get
// right — every case of ``lucid/test/unit/compile/_grad_matrix.py`` that
// uses them matches eager backward with the manual VJPs switched off.
// Outside this list ``gradientForPrimaryTensor:`` either aborts the process
// ("Couldn't get gradient Tensor" — every select on a comparison: where,
// masked_fill, leaky_relu, elu, softplus, …; max / min, the cumulative ops,
// sort, prod, tile, det, erfinv), which no caller can catch, or returns a
// wrong gradient (arcsin, reciprocal, triu / tril off the diagonal).  So the
// autodiff fallback is taken only for a trace made entirely of these, and
// anything else fails the compile and runs eager.  ``tril`` and
// ``broadcast_to`` are left out although their measured cases passed: an
// off-diagonal ``tril`` and a ``broadcast_to`` of a slice did not.
static const std::unordered_set<std::string_view> kAutodiffSafeOps{
    "abs",
    "add",
    "arange",
    "arccos",
    "arctan",
    "astype",
    "avg_pool2d",
    "batch_norm_eval",
    "bilinear_layer",
    "clip",
    "concatenate",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "cos",
    "cosh",
    "diagonal",
    "div",
    "dot",
    "dropout",
    "einsum",
    "erf",
    "exp",
    "eye",
    "flip",
    "floor",
    "full",
    "gather",
    "gelu",
    "gelu_exact",
    "group_norm",
    "hard_sigmoid",
    "hard_swish",
    "inner",
    "inv",
    "layer_norm",
    "linear",
    "log",
    "log2",
    "log_softmax",
    "lp_normalize",
    "matmul",
    "max_pool2d",
    "maximum",
    "mean",
    "minimum",
    "mish",
    "mse_loss",
    "mul",
    "neg",
    "not_equal",
    "outer",
    "pad",
    "permute",
    "pow",
    "rand",
    "randn",
    "relu",
    "relu6",
    "reshape",
    "rms_norm",
    "roll",
    "rsqrt",
    "scaled_dot_product_attention",
    "scatter_add",
    "sigmoid",
    "silu",
    "sin",
    "sinh",
    "softmax",
    "split",
    "split_at",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "tan",
    "tanh",
    "tensordot",
    "trace",
    "unbind",
    "unfold_dim",
    "unsqueeze",
    "var",
    "zeros",
};

inline std::string first_op_unsafe_for_autodiff(const TraceGraph& graph) {
    if (!graph.vjp_regions.empty())
        return "custom_vjp";
    for (const auto& node : graph.ops)
        if (kAutodiffSafeOps.find(node.name) == kAutodiffSafeOps.end())
            return node.name;
    return {};
}

inline std::string first_invalid_graph_op(const TraceGraph& graph,
                                          Device device,
                                          bool allow_host_const,
                                          bool check_vjp_regions) {
    auto check_list = [&](const std::vector<OpNode>& ops, bool skip_empty) -> std::string {
        for (const auto& op : ops) {
            if (op.outputs.empty() || op.outputs[0].device != device)
                return "mixed-device trace";
            for (TensorId iid : op.inputs) {
                if (iid < 0)
                    return "op '" + op.name + "' has an unresolved input slot";
            }
            if (find_emitter(op.name) == nullptr) {
                if (skip_empty && op.inputs.empty() &&
                    (!allow_host_const || result_is_a_host_constant(op.name)))
                    continue;
                return "no emitter registered for op '" + op.name + "'";
            }
        }
        return {};
    };
    if (std::string err = check_list(graph.ops, true); !err.empty())
        return err;
    if (check_vjp_regions) {
        for (const auto& r : graph.vjp_regions) {
            if (std::string err = check_list(r.bwd_ops, true); !err.empty())
                return err;
        }
    }
    return {};
}

inline void mark_vjp_regions_consumed(const TraceGraph& graph,
                                      std::unordered_set<TensorId>& trace_consumed) {
    for (const auto& r : graph.vjp_regions) {
        for (const auto& om : r.output_metas)
            if (om.id >= 0)
                trace_consumed.insert(om.id);
        for (TensorId cid : r.cotangent_ids)
            if (cid >= 0)
                trace_consumed.insert(cid);
        for (TensorId gid : r.input_grad_ids)
            if (gid >= 0)
                trace_consumed.insert(gid);
        for (const auto& n : r.bwd_ops)
            for (TensorId iid : n.inputs)
                if (iid >= 0)
                    trace_consumed.insert(iid);
    }
}

inline NSArray<NSNumber*>* shape_to_nsarray(const Shape& shape) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (std::int64_t d : shape)
        [out addObject:[NSNumber numberWithLongLong:d]];
    return out;
}

// Builds a compile descriptor.
//
//   * Determinism ON (uncommon path): ``optimizationLevel = Level0``
//     to guarantee bit-stable kernel selection across runs.
//   * Determinism OFF (default): ``optimizationLevel = Level1`` — pay
//     extra compile-time analysis for runtime fusion wins.  On macOS
//     26 SDK this opens up kernel-merge passes MPSGraph keeps gated
//     behind Level1; the cost is a one-time ~5-20ms compile-time
//     increase per signature, amortised over every subsequent run.
//
//   * Half precision: ``Level0`` too.  Level1 also places ops across
//     devices, the Neural Engine included, and on M4-class hardware a
//     float16 graph placed there comes back wrong — a float16 Linear under
//     autocast returned 25152 where eager returned 0.0116, with MPSGraph
//     printing "Incompatible element type for ANE".  Level0 keeps the
//     graph on the GPU; float32 graphs keep Level1, which is correct there.
inline MPSGraphCompilationDescriptor* make_compile_descriptor(bool half_precision) {
    MPSGraphCompilationDescriptor* desc = [[MPSGraphCompilationDescriptor alloc] init];
    if (::lucid::Determinism::is_enabled() || half_precision)
        desc.optimizationLevel = MPSGraphOptimizationLevel0;
    else
        desc.optimizationLevel = MPSGraphOptimizationLevel1;
    return desc;
}

// Whether any value the trace computes is float16 or bfloat16 — the graphs
// make_compile_descriptor keeps off the Neural Engine.
template <class Graph>
bool carries_half_precision(const Graph& graph) {
    for (const auto& node : graph.ops)
        for (const auto& meta : node.outputs)
            if (meta.dtype == Dtype::F16 || meta.dtype == Dtype::BF16)
                return true;
    for (const auto& r : graph.vjp_regions)
        for (const auto& node : r.bwd_ops)
            for (const auto& meta : node.outputs)
                if (meta.dtype == Dtype::F16 || meta.dtype == Dtype::BF16)
                    return true;
    return false;
}

// Collect the trace ids that carry attention weights — a ``softmax`` output,
// or a value transitively derived from one through an (inference-mode identity)
// ``dropout`` (the ``softmax → attn_drop → @V`` pattern of ViT / GPT-2).  A
// matmul that reads such an id is an attention value-projection; the
// MatmulEmitter emits those transposed so MPSGraph does not pattern-match
// ``softmax(...) @ V`` onto its fused-attention kernel, which silently
// miscompiles for some shapes (sequence length in [17,24] with batch>=3 on
// macOS 26 — see the engine note).  Ops are in topological order, so one
// forward pass propagates the marker correctly.
inline std::unordered_set<TensorId> collect_softmax_outputs(const TraceGraph& graph) {
    std::unordered_set<TensorId> derived;
    auto input_is_derived = [&](const OpNode& n) {
        for (TensorId iid : n.inputs)
            if (iid >= 0 && derived.count(iid) != 0)
                return true;
        return false;
    };
    auto scan = [&](const std::vector<OpNode>& nodes) {
        for (const auto& n : nodes) {
            if (n.name == "softmax") {
                for (const auto& meta : n.outputs)
                    derived.insert(meta.id);
            } else if (n.name == "dropout" && input_is_derived(n)) {
                for (const auto& meta : n.outputs)
                    derived.insert(meta.id);
            }
        }
    };
    scan(graph.ops);
    for (const auto& r : graph.vjp_regions)
        scan(r.bwd_ops);
    return derived;
}

}  // namespace

// Attention fused-pass workaround capability flag (declared in OpEmitter.h).
// Plain int: written once by the Python probe before real attention compiles,
// then only read by the emitters — a benign relaxed access, no lock needed.
// Defaults to -1 (unprobed) so the emitters apply the always-correct transpose
// until the probe affirmatively clears this GPU.
static int g_attention_workaround_state = -1;

int attention_workaround_state() {
    return g_attention_workaround_state;
}
void set_attention_workaround_state(int s) {
    g_attention_workaround_state = s;
}

}  // namespace lucid::compile

// Bring the .mm-only class definition into scope.  CompiledExecutable
// is fully defined inside the other translation unit; here we redeclare
// it as a complete class to access its fields (one TU pattern).
namespace lucid::compile {

// Full definition that matches CompiledExecutable.mm.  The two .mm
// files must keep these in sync — they are compiled into separate
// object files but expose the same layout, so the cache + builder can
// read/write the same fields without breaking the ABI.
class CompiledExecutable {
public:
    MPSGraphExecutable* executable = nil;  // ARC strong (matches .mm)
    // Phase 1.9: retained source MPSGraph for variable-bearing
    // executables — keeps the variable storage live for the executable's
    // lifetime.  See the matching field in CompiledExecutable.mm for
    // rationale.  nullptr for non-variable compiles (the common case).
    void* source_graph = nullptr;  // __bridge_retained MPSGraph*
    std::vector<TensorId> input_ids;
    std::vector<TensorId> output_ids;
    std::vector<Shape> input_shapes;
    std::vector<Dtype> input_dtypes;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;
    Device device = Device::GPU;
    std::vector<TensorId> grad_output_ids;              // Phase 1.3
    bool dynamic_batch = false;                         // Phase 1.6
    std::unordered_set<std::size_t> static_feed_slots;  // Phase 1.6
    // Bind order against ``executable.feedTensors``: position ``k`` of the
    // ``inputsArray`` handed to MPSGraph carries Lucid feed slot
    // ``feed_order[k]``.  Empty = the two orders coincide, which is the
    // case for a freshly compiled executable.  A package round-trip does
    // NOT preserve the order (the graph is compiled from an NSDictionary
    // of feeds, so the order was never ours to choose), which is why
    // ``load_executable`` re-derives this from the placeholder names.
    std::vector<std::size_t> feed_order;

    ~CompiledExecutable() {
        // Mirror the destructor in CompiledExecutable.mm — release the
        // retained source MPSGraph (if any).  Defined inline here so the
        // C++ destructor calls the right release; ARC handles
        // ``executable``.
        if (source_graph != nullptr) {
            @autoreleasepool {
                MPSGraph* g = (__bridge_transfer MPSGraph*)source_graph;
                (void)g;
            }
            source_graph = nullptr;
        }
    }
};

}  // namespace lucid::compile

namespace lucid::compile {

// ────────────────────────────────────────────────────────────────────
// MpsBuilder lifecycle (Phase B of compile OOP refactor)
// ────────────────────────────────────────────────────────────────────
MpsBuilder::MpsBuilder(const TraceGraph& graph,
                       const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                       std::string* error_msg)
    : graph_(graph), external_feeds_(external_feeds), error_msg_(error_msg) {}

MpsBuilder::~MpsBuilder() = default;

CompiledExecutable* MpsBuilder::compile_trace(bool dynamic_batch,
                                              const std::vector<TensorId>& param_ids,
                                              const std::vector<TensorId>& explicit_outputs) {
    auto& graph = graph_;
    const auto& external_feeds = external_feeds_;
    auto* error_msg = error_msg_;

    // ``dynamic_batch`` (Phase 1.6): leading dim of every non-parameter
    // feed becomes a symbolic placeholder (-1) so a single executable
    // handles variable batch size.  ``param_ids`` supplies which feed
    // ids are model parameters whose first dim is structural (not a
    // batch axis) — those stay fixed.  The forward-only path can do
    // this safely because it does not call ``gradientForPrimaryTensor:``,
    // which currently refuses to build gradients for dynamic-shape
    // primaries.  ``make_step`` / the backward-aware path therefore
    // continues to compile a separate executable per batch size.
    const std::unordered_set<TensorId> param_id_set(param_ids.begin(), param_ids.end());
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_trace: empty TraceGraph");

    // Trace must be single-device — Phase 1.2 supports GPU only.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (op.outputs.empty())
            continue;
        device = op.outputs[0].device;
        break;
    }
    if (device != Device::GPU)
        return fail("compile_trace: only Device::GPU traces are supported in Phase 1.2");

    // Every op must have all of its inputs resolved (no -1 sentinels)
    // and must have an emitter registered.  These are the two reasons
    // we fall back to eager at this stage.
    //
    // Exception: a host-precomputed factory (arange / eye / linspace /
    // logspace / meshgrid …) records an :class:`OpScopeFull` but
    // doesn't wire its output through the tracer.  Downstream consumers
    // therefore see the factory's :type:`TensorImpl` as a *fresh
    // external feed*, not an intermediate, and the op header in the
    // trace is dead code.  We skip the emitter check for such
    // ``inputs.empty()`` ops; the emit loop below mirrors the skip.
    if (const std::string empty = first_empty_tensor(graph, external_feeds); !empty.empty())
        return fail("compile_trace: zero-size tensor at " + empty + " — MPSGraph aborts on one");
    if (const std::string bad_op = first_invalid_graph_op(graph, device, true, false);
        !bad_op.empty())
        return fail("compile_trace: " + bad_op);

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Build placeholders for every external feed.  The id order is
        // the order MPSGraph will expect inputs in at run time — we
        // sort by id for determinism (so two compiled traces with the
        // same structure always have the same feed order regardless
        // of map iteration order).
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_trace: external feed for id " + std::to_string(tid) +
                            " is null");
            if (impl->device() != device)
                return fail("compile_trace: external feed device mismatch");

            const Shape feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            const bool is_user_input_ct = (param_id_set.find(tid) == param_id_set.end());
            NSArray<NSNumber*>* ns_shape;
            if (dynamic_batch && is_user_input_ct && !feed_shape.empty()) {
                NSMutableArray<NSNumber*>* dyn =
                    [NSMutableArray arrayWithCapacity:feed_shape.size()];
                [dyn addObject:[NSNumber numberWithLongLong:-1]];
                for (std::size_t k = 1; k < feed_shape.size(); ++k)
                    [dyn addObject:[NSNumber numberWithLongLong:feed_shape[k]]];
                ns_shape = dyn;
            } else {
                ns_shape = shape_to_nsarray(feed_shape);
            }
            MPSDataType ns_dt;
            try {
                ns_dt = mps_dtype_or_throw(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_trace: ") + e.what());
            }

            MPSGraphTensor* ph = [graph_obj
                placeholderWithShape:ns_shape
                            dataType:ns_dt
                                name:[NSString stringWithFormat:@"feed_%lld", (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Verbose: when LUCID_COMPILE_VERBOSE=1 print per-op input/output
        // MPSGraph tensor shapes so reshape-compat failures can be
        // localised without external profilers.
        const bool verbose = []() {
            const char* s = std::getenv("LUCID_COMPILE_VERBOSE");
            return s && *s == '1';
        }();

        // Pre-compute the trace-wide "consumed by some op" set and hand
        // it to BuilderContext so multi-output emitters (split / split_at
        // / unbind / chunk …) can skip binding pieces that no downstream
        // op uses.  Without this filter, dead pieces would silently
        // appear as additional graph outputs (a single ``a, _ = x.split``
        // becomes a 2-output executable returning ``_`` to the user).
        // Liveness set for multi-output emitters (split / split_at / unbind /
        // topk / lstm): an id is "consumed" iff it is needed to produce a graph
        // output.  When the caller supplied ``explicit_outputs`` we compute
        // TRUE liveness by walking backward from those outputs through each
        // producing op's inputs.  This (a) marks a *returned* piece live even
        // when no later op consumes it (``return x[1:2]``), and (b) recognises
        // a piece that flows only into a discarded result as DEAD — e.g. an
        // LSTM's ``c_n`` that feeds a ``cat`` the user threw away — so the
        // emitter can skip / bail on it.  Without explicit outputs (legacy
        // single-output path) fall back to the looser "consumed by some op".
        std::unordered_set<TensorId> trace_consumed;
        if (!explicit_outputs.empty()) {
            std::unordered_map<TensorId, const OpNode*> producer;
            for (const auto& n : graph.ops)
                for (const auto& meta : n.outputs)
                    producer.emplace(meta.id, &n);
            std::vector<TensorId> work;
            for (TensorId oid : explicit_outputs)
                if (oid >= 0 && trace_consumed.insert(oid).second)
                    work.push_back(oid);
            while (!work.empty()) {
                const TensorId t = work.back();
                work.pop_back();
                const auto it = producer.find(t);
                if (it == producer.end())
                    continue;  // external feed / placeholder — no producer
                for (TensorId iid : it->second->inputs)
                    if (iid >= 0 && trace_consumed.insert(iid).second)
                        work.push_back(iid);
            }
        } else {
            for (const auto& n : graph.ops)
                for (TensorId iid : n.inputs)
                    if (iid >= 0)
                        trace_consumed.insert(iid);
        }
        ctx.set_consumed_inputs(trace_consumed);
        ctx.set_softmax_outputs(collect_softmax_outputs(graph));

        // Emit ops in dispatch order.
        std::size_t op_idx = 0;
        for (const auto& node : graph.ops) {
            // Dead-op elimination (liveness known ⇒ explicit_outputs given):
            // skip an op whose every output is dead.  It can't reach a graph
            // output, and emitting it can force a crash-prone kernel — e.g. an
            // LSTM cell trajectory (``produceCell``) feeding a discarded
            // ``cat`` aborts the Metal LSTM kernel on some drivers.
            if (!explicit_outputs.empty() && !node.outputs.empty()) {
                bool any_live = false;
                for (const auto& meta : node.outputs)
                    if (trace_consumed.count(meta.id) != 0) {
                        any_live = true;
                        break;
                    }
                if (!any_live) {
                    // Recorded inputs ⇒ the op traced normally and its
                    // result is genuinely unread, so dropping it frees a
                    // kernel and changes nothing.  No inputs ⇒ either a
                    // host constant, or an op that never recorded its
                    // I/O and whose value is about to be frozen into the
                    // graph as a feed.  Refuse the second.
                    if (node.inputs.empty() && !result_is_a_host_constant(node.name))
                        return fail("compile_trace: op '" + node.name +
                                    "' did not record its trace I/O — its result would be "
                                    "frozen into the graph at trace time");
                    ++op_idx;
                    continue;
                }
            }

            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty()) {
                    // Dead-code host-factory header (arange / eye / …)
                    // — its output already shows up as an external feed
                    // through whatever consumer used the tensor next.
                    ++op_idx;
                    continue;
                }
                // Defensive against a racy registry mutation between
                // the precheck above and this loop.
                return fail("compile_trace: emitter vanished for op '" + node.name + "'");
            }

            if (verbose) {
                fprintf(stderr, "[compile] op[%zu] %s  inputs=[", op_idx, node.name.c_str());
                for (std::size_t i = 0; i < node.inputs.size(); ++i) {
                    if (i)
                        fputc(',', stderr);
                    TensorId iid = node.inputs[i];
                    if (iid < 0) {
                        fputs("none", stderr);
                        continue;
                    }
                    MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(iid);
                    if (t == nil) {
                        fprintf(stderr, "id%lld[unresolved]", (long long)iid);
                        continue;
                    }
                    fputc('(', stderr);
                    NSArray<NSNumber*>* sh = t.shape;
                    for (NSUInteger k = 0; k < sh.count; ++k) {
                        if (k)
                            fputc(',', stderr);
                        fprintf(stderr, "%lld", [sh[k] longLongValue]);
                    }
                    fputc(')', stderr);
                }
                fputs("] expected_out=(", stderr);
                if (!node.outputs.empty()) {
                    const auto& sh = node.outputs[0].shape;
                    for (std::size_t k = 0; k < sh.size(); ++k) {
                        if (k)
                            fputc(',', stderr);
                        fprintf(stderr, "%lld", (long long)sh[k]);
                    }
                }
                fputs(")\n", stderr);
            }

            if (const std::string why = emit_node(*emitter, ctx, node); !why.empty())
                return fail("compile_trace: " + why);
            // Emitters bind their own outputs explicitly via
            // ``ctx.bind(outputs[k].id, ...)`` — no auto-bind here.
            if (verbose && !node.outputs.empty()) {
                fprintf(stderr, "[compile]   → emitted (");
                const auto& sh = node.outputs[0].shape;
                for (std::size_t k = 0; k < sh.size(); ++k) {
                    if (k)
                        fputc(',', stderr);
                    fprintf(stderr, "%lld", (long long)sh[k]);
                }
                fputs(")\n", stderr);
                fflush(stderr);
            }
            ++op_idx;
        }

        // Build the target list: any output id that is not consumed by
        // a later op is a graph output.  Phase 1.2 step 1 typically
        // has a single Linear → single target.
        std::unordered_set<TensorId> consumed;
        for (const auto& node : graph.ops)
            for (TensorId iid : node.inputs)
                if (iid >= 0)
                    consumed.insert(iid);

        std::vector<TensorId> target_ids;
        std::vector<MPSGraphTensor*> target_tensors;
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;

        auto add_target = [&](TensorId id, const Shape& shape, Dtype dtype) -> bool {
            void* t_void = ctx.resolve(id);
            if (t_void == nullptr)
                return false;
            target_ids.push_back(id);
            target_tensors.push_back((__bridge MPSGraphTensor*)t_void);
            output_shapes.push_back(shape);
            output_dtypes.push_back(dtype);
            return true;
        };

        if (!explicit_outputs.empty()) {
            // Caller specified exactly which ids are graph outputs (e.g.
            // the Python compile harness extracted them from the user's
            // return value).  Look up each id in the trace metadata and
            // bind it as a target.  Honoring this list drops "unconsumed
            // but Python-discarded" intermediates (RNN's ``h_n`` when
            // user does ``out, _ = rnn(x)``).
            std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
            for (const auto& node : graph.ops)
                for (const auto& meta : node.outputs)
                    id_to_meta[meta.id] = {meta.shape, meta.dtype};
            // External feeds can also be returned (`return x` directly).
            // Look those up from the feed map; the placeholder is already
            // bound in ctx.
            for (TensorId tid : explicit_outputs) {
                auto it = id_to_meta.find(tid);
                if (it != id_to_meta.end()) {
                    if (!add_target(tid, it->second.first, it->second.second))
                        return fail("compile_trace: explicit_output id " + std::to_string(tid) +
                                    " has no MPSGraph binding");
                } else {
                    // Try external feed.
                    auto fit = external_feeds.find(tid);
                    if (fit == external_feeds.end() || !fit->second)
                        return fail("compile_trace: explicit_output id " + std::to_string(tid) +
                                    " not in trace");
                    if (!add_target(tid, fit->second->shape(), fit->second->dtype()))
                        return fail("compile_trace: explicit_output id " + std::to_string(tid) +
                                    " external feed not bound");
                }
            }
        } else {
            // Auto-detect: walk every output slot of each op, skip ones
            // consumed by a later op, skip un-bound (dead-piece / host
            // factory) ids.  The emitter for multi-output ops handles
            // ctx.bind selectively so dead pieces resolve to nullptr.
            for (const auto& node : graph.ops) {
                for (const auto& meta : node.outputs) {
                    if (consumed.count(meta.id))
                        continue;
                    if (!add_target(meta.id, meta.shape, meta.dtype)) {
                        // Either a host-side factory header (arange /
                        // eye / linspace …) or an un-bound dead piece
                        // of a multi-output op.  Drop silently.
                        continue;
                    }
                }
            }
        }
        if (target_tensors.empty())
            return fail("compile_trace: no graph outputs found (every op output consumed)");

        // Build the feed dictionary for compileWithDevice:feeds: — maps
        // each placeholder to a shape descriptor.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = mps_dtype_or_throw(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            MPSGraphShapedType* st = [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                                      dataType:ns_dt];
            feed_dict[feed_tensors[i]] = st;
        }
        NSMutableArray<MPSGraphTensor*>* target_arr =
            [NSMutableArray arrayWithCapacity:target_tensors.size()];
        for (MPSGraphTensor* t : target_tensors)
            [target_arr addObject:t];

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled =
            [graph_obj compileWithDevice:mps_device
                                   feeds:feed_dict
                           targetTensors:target_arr
                        targetOperations:nil
                   compilationDescriptor:make_compile_descriptor(carries_half_precision(graph))];
        if (compiled == nil)
            return fail("compile_trace: MPSGraph compilation returned nil");

        auto exe = std::make_unique<CompiledExecutable>();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = target_ids;
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        // Phase 1.6: tell the runner which feed slots are parameters
        // (whose first dim is structural) so a BS != trace-BS call
        // only rewrites the leading axis for non-parameter slots.
        if (dynamic_batch) {
            exe->dynamic_batch = true;
            const std::unordered_set<TensorId> param_id_set_for_exe(param_ids.begin(),
                                                                    param_ids.end());
            for (std::size_t i = 0; i < exe->input_ids.size(); ++i) {
                if (param_id_set_for_exe.find(exe->input_ids[i]) != param_id_set_for_exe.end()) {
                    exe->static_feed_slots.insert(i);
                }
            }
        }
        return exe.release();
    }
}

// ── Phase 1.3 ────────────────────────────────────────────────────────────────

CompiledExecutable*
MpsBuilder::compile_trace_with_backward(TensorId loss_id,
                                        const std::vector<TensorId>& param_ids,
                                        bool dynamic_batch,
                                        const std::vector<TensorId>& extra_output_ids) {
    auto& graph = graph_;
    const auto& external_feeds = external_feeds_;
    auto* error_msg = error_msg_;

    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_trace_with_backward: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_trace_with_backward: param_ids must be non-empty");

    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (op.outputs.empty())
            continue;
        device = op.outputs[0].device;
        break;
    }
    if (device != Device::GPU)
        return fail("compile_trace_with_backward: only Device::GPU traces are supported");

    if (const std::string empty = first_empty_tensor(graph, external_feeds); !empty.empty())
        return fail("compile_trace_with_backward: zero-size tensor at " + empty +
                    " — MPSGraph aborts on one");
    if (const std::string bad_op = first_invalid_graph_op(graph, device, true, true);
        !bad_op.empty())
        return fail("compile_trace_with_backward: " + bad_op);

    // AMP / mixed-dtype support (X4 of compile-completeness plan).
    //
    // The historical precheck rejected any trace whose floating
    // dtypes mixed (e.g. F16 forward + F32 reductions) because
    // MPSGraph's ``gradientForPrimaryTensor:`` would abort with an
    // incompatible-broadcast error on the auto-derived backward.
    //
    // The manual VJP path (LUCID_MANUAL_VJP, defaulted ON in P7)
    // doesn't use ``gradientForPrimaryTensor:`` for the ops it
    // covers — it emits the bwd subgraph directly with explicit
    // dtype handling per VJP.  Every VJP's ``constantWithScalar:``
    // calls already use the input tensor's dtype, and the manual-VJP
    // F16 smoke test (P7b) confirms dtype propagation works end-to-end.
    //
    // We therefore lift the mixed-dtype rejection.  When manual VJP
    // is OFF and the trace falls back to ``gradientForPrimaryTensor:``
    // on a heterogeneous-dtype trace, MPSGraph may still abort —
    // that's the legacy behaviour and a known limitation of the
    // autograd fallback path.  Documentation should steer AMP users
    // toward the manual VJP path (which is the default).
    //
    // Integral / Bool dtypes are exempt from the AMP discussion —
    // they live in CE's gather / mask paths and never feed into the
    // backward graph.

    // Every param id must be an external feed (gradient is meaningful only
    // for trace inputs; intermediate-tensor gradients are not user state).
    for (TensorId pid : param_ids) {
        if (external_feeds.find(pid) == external_feeds.end())
            return fail("compile_trace_with_backward: param id " + std::to_string(pid) +
                        " is not an external feed");
    }

    // Phase 1.6: ``dynamic_batch`` opts the leading dim of every
    // non-parameter, non-structural feed into a symbolic placeholder
    // (-1).  Parameter shapes (weights, biases) stay fixed because
    // their first dim is structural (out_channels, etc.) and not a
    // batch axis.
    std::unordered_set<TensorId> param_id_set(param_ids.begin(), param_ids.end());

    // 3.5: BatchNorm running-stat feeds (the (C,) running_mean / running_var
    // fed into a 5-input BN-train node) are ALSO structural — their dim 0 is
    // the channel count, not batch — and they are buffers, not params, so they
    // are not in param_id_set.  Collect them so the dynamic-batch rewrite below
    // does NOT symbolise their channel axis to -1.
    std::unordered_set<TensorId> static_feed_ids;
    if (dynamic_batch) {
        for (const auto& node : graph.ops) {
            if ((node.name == "batch_norm" || node.name == "batch_norm1d" ||
                 node.name == "batch_norm3d") &&
                node.inputs.size() >= 5) {
                if (node.inputs[3] >= 0)
                    static_feed_ids.insert(node.inputs[3]);
                if (node.inputs[4] >= 0)
                    static_feed_ids.insert(node.inputs[4]);
            }
        }
    }

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_trace_with_backward: feed id " + std::to_string(tid) +
                            " is null");
            if (impl->device() != device)
                return fail("compile_trace_with_backward: feed device mismatch");

            const Shape feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            const bool is_user_input = (param_id_set.find(tid) == param_id_set.end()) &&
                                       (static_feed_ids.find(tid) == static_feed_ids.end());
            NSArray<NSNumber*>* ns_shape;
            if (dynamic_batch && is_user_input && !feed_shape.empty()) {
                NSMutableArray<NSNumber*>* dyn =
                    [NSMutableArray arrayWithCapacity:feed_shape.size()];
                [dyn addObject:[NSNumber numberWithLongLong:-1]];
                for (std::size_t k = 1; k < feed_shape.size(); ++k)
                    [dyn addObject:[NSNumber numberWithLongLong:feed_shape[k]]];
                ns_shape = dyn;
            } else {
                ns_shape = shape_to_nsarray(feed_shape);
            }
            MPSDataType ns_dt;
            try {
                ns_dt = mps_dtype_or_throw(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_trace_with_backward: ") + e.what());
            }
            MPSGraphTensor* ph = [graph_obj
                placeholderWithShape:ns_shape
                            dataType:ns_dt
                                name:[NSString stringWithFormat:@"feed_%lld", (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Pre-compute trace-wide consumed set — multi-output emitters
        // (split / split_at / unbind / chunk / topk / lstm) need this
        // to know which pieces to bind.  See parallel block in
        // compile_trace + compile_generic_fused_step.
        {
            std::unordered_set<TensorId> trace_consumed;
            for (const auto& n : graph.ops)
                for (TensorId iid : n.inputs)
                    if (iid >= 0)
                        trace_consumed.insert(iid);
            trace_consumed.insert(loss_id);
            // 3.5: explicit non-gradient outputs (e.g. BN running-stat EMA) are
            // consumed too, so their multi-output emitter binds outputs[1]/[2].
            for (TensorId tid : extra_output_ids)
                if (tid >= 0)
                    trace_consumed.insert(tid);
            mark_vjp_regions_consumed(graph, trace_consumed);
            ctx.set_consumed_inputs(trace_consumed);
            ctx.set_softmax_outputs(collect_softmax_outputs(graph));
        }

        // Emit forward ops.
        for (const auto& node : graph.ops) {
            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty() && result_is_a_host_constant(node.name))
                    continue;  // host constant — see compile_trace
                return fail("compile_trace_with_backward: emitter vanished for op '" + node.name +
                            "'");
            }
            if (const std::string why = emit_node(*emitter, ctx, node); !why.empty())
                return fail("compile_trace_with_backward: " + why);
            // Emitters bind their own outputs via ctx.bind() — no auto-bind here.
        }

        // Resolve the loss tensor.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_trace_with_backward: loss id " + std::to_string(loss_id) +
                        " has no MPSGraph tensor (not produced by trace?)");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        // Note: MPSGraph's gradientForPrimaryTensor mishandles 0-D
        // scalar primaries.  The Reduction emitter (OpEmitters/Reduction.mm)
        // works around this by leaving one size-1 axis when reducing
        // every dim, so any loss that originated as a full-reduce sum/mean
        // is at least 1-D from MPSGraph's point of view (output_shapes
        // still records the user-visible 0-D shape so the buffer wrap
        // matches eager exactly).

        // Resolve param placeholders.
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_trace_with_backward: param id " + std::to_string(pid) +
                            " has no placeholder");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Auto-derive gradients.  Manual VJP path opt-in via env var;
        // falls through to MPSGraph autograd on coverage gap unless
        // LUCID_MANUAL_VJP_REQUIRE=1.
        std::vector<MPSGraphTensor*> grad_tensors;
        std::vector<Shape> grad_shapes;
        std::vector<Dtype> grad_dtypes;
        grad_tensors.reserve(param_ids.size());
        grad_shapes.reserve(param_ids.size());
        grad_dtypes.reserve(param_ids.size());

        // Per parameter: traced, but the loss does not depend on it.  Eager
        // leaves that gradient None; its grad id below is -1 so the caller
        // can too, and the executable fills the slot with zeros.
        std::vector<bool> absent_grads(param_ids.size(), false);
        bool grads_done = false;
        std::string vjp_err;
        {
            std::vector<void*> grads_void;
            switch (try_manual_vjp_grads((__bridge void*)graph_obj, ctx, graph, loss_id, param_ids,
                                         grads_void, &vjp_err, &absent_grads)) {
            case ManualVjpStatus::Success:
                for (std::size_t i = 0; i < param_ids.size(); ++i) {
                    grad_tensors.push_back((__bridge MPSGraphTensor*)grads_void[i]);
                    const auto& p_impl = external_feeds.at(param_ids[i]);
                    grad_shapes.push_back(p_impl->shape());
                    grad_dtypes.push_back(p_impl->dtype());
                }
                grads_done = true;
                break;
            case ManualVjpStatus::HardFail:
                return fail("compile_trace_with_backward: "
                            "LUCID_MANUAL_VJP_REQUIRE=1 but manual VJP gap — " +
                            vjp_err);
            case ManualVjpStatus::FellBack:
            case ManualVjpStatus::Disabled:
                break;  // → MPSGraph autograd below
            }
        }

        if (!grads_done) {
            if (const std::string bad = first_op_unsafe_for_autodiff(graph); !bad.empty())
                return fail("compile_trace_with_backward: " + vjp_err + "; op '" + bad +
                            "' rules out MPSGraph's autodiff as the fallback");
            // Ask only for the parameters the loss depends on — MPSGraph
            // aborts on any other rather than returning nil.  The rest get
            // the zero, absent gradient below.
            const auto ancestors = loss_ancestors(graph, loss_id);
            NSMutableArray<MPSGraphTensor*>* reach_arr =
                [NSMutableArray arrayWithCapacity:param_ids.size()];
            for (std::size_t i = 0; i < param_ids.size(); ++i)
                if (ancestors.count(param_ids[i]) != 0)
                    [reach_arr addObject:param_arr[i]];
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                reach_arr.count == 0 ? @{}
                                     : [graph_obj gradientForPrimaryTensor:loss_t
                                                               withTensors:reach_arr
                                                                      name:@"lucid_grads"];
            if (grad_map == nil)
                return fail("compile_trace_with_backward: gradientForPrimaryTensor returned nil");

            for (std::size_t i = 0; i < param_ids.size(); ++i) {
                MPSGraphTensor* g_t = grad_map[param_arr[i]];
                if (g_t == nil) {
                    // Not on the loss path — as in the manual walk above.
                    MPSGraphTensor* p_t = param_arr[i];
                    g_t = [graph_obj constantWithScalar:0.0 shape:p_t.shape dataType:p_t.dataType];
                    absent_grads[i] = true;
                }
                grad_tensors.push_back(g_t);
                const auto& p_impl = external_feeds.at(param_ids[i]);
                grad_shapes.push_back(p_impl->shape());
                grad_dtypes.push_back(p_impl->dtype());
            }
        }

        // Targets = [loss] + grad tensors.
        NSMutableArray<MPSGraphTensor*>* target_arr =
            [NSMutableArray arrayWithCapacity:1 + grad_tensors.size()];
        [target_arr addObject:loss_t];
        for (MPSGraphTensor* g_t : grad_tensors)
            [target_arr addObject:g_t];

        // 3.5: explicit extra outputs (BN running-stat EMA) come AFTER the grads
        // in target order, so the runtime returns [loss, *grads, *extra].  Their
        // shape/dtype meta is read from the trace IR (the emitter bound their
        // MPSGraph tensors above; resolve them now).
        std::vector<Shape> extra_shapes;
        std::vector<Dtype> extra_dtypes;
        extra_shapes.reserve(extra_output_ids.size());
        extra_dtypes.reserve(extra_output_ids.size());
        if (!extra_output_ids.empty()) {
            std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
            for (const auto& node : graph.ops)
                for (const auto& meta : node.outputs)
                    id_to_meta[meta.id] = {meta.shape, meta.dtype};
            for (TensorId tid : extra_output_ids) {
                void* tv = ctx.resolve(tid);
                if (tv == nullptr)
                    return fail("compile_trace_with_backward: extra output id " +
                                std::to_string(tid) + " has no MPSGraph binding");
                auto it = id_to_meta.find(tid);
                if (it == id_to_meta.end())
                    return fail("compile_trace_with_backward: extra output id " +
                                std::to_string(tid) + " not in trace");
                [target_arr addObject:(__bridge MPSGraphTensor*)tv];
                extra_shapes.push_back(it->second.first);
                extra_dtypes.push_back(it->second.second);
            }
        }

        // Build feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = mps_dtype_or_throw(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] = [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                                          dataType:ns_dt];
        }

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled =
            [graph_obj compileWithDevice:mps_device
                                   feeds:feed_dict
                           targetTensors:target_arr
                        targetOperations:nil
                   compilationDescriptor:make_compile_descriptor(carries_half_precision(graph))];
        if (compiled == nil)
            return fail("compile_trace_with_backward: MPSGraph compilation returned nil");

        // Mint fresh ids for the grad outputs, well past the trace's
        // id space.  Convention: grad_id_for_param[i] = graph.next_id + 1 + i
        // (offset by 1 to leave room for any future loss-id alias).
        std::vector<TensorId> grad_output_ids;
        grad_output_ids.reserve(param_ids.size() + extra_output_ids.size());
        TensorId next_grad_id = graph.next_id + 1;
        for (std::size_t i = 0; i < param_ids.size(); ++i)
            grad_output_ids.push_back(
                absent_grads[i] ? TensorId{-1} : TensorId{next_grad_id + static_cast<TensorId>(i)});
        // 3.5: explicit extra outputs reuse their trace ids and trail the grads
        // in the runtime's target order ([loss, *grads, *extra]).
        for (TensorId tid : extra_output_ids)
            grad_output_ids.push_back(tid);

        // Assemble the per-output meta vectors in target order: loss first,
        // then grads in param_ids order.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        output_shapes.reserve(1 + grad_shapes.size());
        output_dtypes.reserve(1 + grad_dtypes.size());

        // Find the loss meta in the trace.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_trace_with_backward: loss id meta missing from trace");
        output_shapes.push_back(std::move(loss_shape));
        output_dtypes.push_back(loss_dtype);
        for (std::size_t i = 0; i < grad_shapes.size(); ++i) {
            output_shapes.push_back(std::move(grad_shapes[i]));
            output_dtypes.push_back(grad_dtypes[i]);
        }
        // 3.5: extra-output meta trails loss + grads, matching target_arr order.
        for (std::size_t i = 0; i < extra_shapes.size(); ++i) {
            output_shapes.push_back(std::move(extra_shapes[i]));
            output_dtypes.push_back(extra_dtypes[i]);
        }

        auto exe = std::make_unique<CompiledExecutable>();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(grad_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe.release();
    }
}

// ── Fused training step (Phase 1.7) ──────────────────────────────────

namespace {

// Build a 0-D constant tensor on ``g`` with the given scalar/dtype.
inline MPSGraphTensor* scalar_const(MPSGraph* g, double v, MPSDataType dt) {
    return [g constantWithScalar:v dataType:dt];
}

// Per-optimizer MPSGraph update emitters.  Each takes the current
// param + grad + state buffers (already resolved as MPSGraphTensors)
// and per-step scalar feeds, and returns (new_param, new_state[]).
//
// SGD with optional momentum / nesterov / weight_decay.  ``mom`` is
// the current momentum tensor (or nullptr if momentum == 0).
struct SgdOutputs {
    MPSGraphTensor* new_param;
    MPSGraphTensor* new_mom;  // nil if no momentum
};

MPSGraphTensor* add_l2_decay(
    MPSGraph* g, MPSGraphTensor* grad, MPSGraphTensor* param, double weight_decay, MPSDataType dt) {
    if (weight_decay == 0.0)
        return grad;
    MPSGraphTensor* wd = scalar_const(g, weight_decay, dt);
    return [g additionWithPrimaryTensor:grad
                        secondaryTensor:[g multiplicationWithPrimaryTensor:wd
                                                           secondaryTensor:param
                                                                      name:nil]
                                   name:nil];
}

SgdOutputs emit_sgd_update(MPSGraph* g,
                           MPSGraphTensor* param,
                           MPSGraphTensor* grad,
                           MPSGraphTensor* mom,
                           const OptimizerSpec& s,
                           MPSDataType dt) {
    MPSGraphTensor* g_eff = add_l2_decay(g, grad, param, s.weight_decay, dt);
    MPSGraphTensor* lr_c = scalar_const(g, s.lr, dt);
    if (s.momentum != 0.0) {
        MPSGraphTensor* mu = scalar_const(g, s.momentum, dt);
        MPSGraphTensor* one_minus_damp = scalar_const(g, 1.0 - s.dampening, dt);
        MPSGraphTensor* new_m =
            [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:mu
                                                            secondaryTensor:mom
                                                                       name:nil]
                         secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_damp
                                                            secondaryTensor:g_eff
                                                                       name:nil]
                                    name:nil];
        MPSGraphTensor* eff_g;
        if (s.nesterov) {
            eff_g = [g additionWithPrimaryTensor:g_eff
                                 secondaryTensor:[g multiplicationWithPrimaryTensor:mu
                                                                    secondaryTensor:new_m
                                                                               name:nil]
                                            name:nil];
        } else {
            eff_g = new_m;
        }
        MPSGraphTensor* new_p =
            [g subtractionWithPrimaryTensor:param
                            secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                               secondaryTensor:eff_g
                                                                          name:nil]
                                       name:nil];
        return {new_p, new_m};
    }
    MPSGraphTensor* new_p = [g subtractionWithPrimaryTensor:param
                                            secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                                               secondaryTensor:g_eff
                                                                                          name:nil]
                                                       name:nil];
    return {new_p, nil};
}

struct AdamOutputs {
    MPSGraphTensor* new_param;
    MPSGraphTensor* new_m;
    MPSGraphTensor* new_v;
};

// Adam / AdamW share the same moment update; the only difference is
// where weight decay is applied: Adam folds it into ``grad`` before
// the moments, AdamW applies it directly to the param outside.
AdamOutputs emit_adam_update(MPSGraph* g,
                             MPSGraphTensor* param,
                             MPSGraphTensor* grad,
                             MPSGraphTensor* m_buf,
                             MPSGraphTensor* v_buf,
                             MPSGraphTensor* bias1,
                             MPSGraphTensor* bias2,
                             const OptimizerSpec& s,
                             MPSDataType dt,
                             bool decoupled_wd) {
    MPSGraphTensor* g_eff = decoupled_wd ? grad : add_l2_decay(g, grad, param, s.weight_decay, dt);
    MPSGraphTensor* beta1_c = scalar_const(g, s.beta1, dt);
    MPSGraphTensor* beta2_c = scalar_const(g, s.beta2, dt);
    MPSGraphTensor* one_minus_b1 = scalar_const(g, 1.0 - s.beta1, dt);
    MPSGraphTensor* one_minus_b2 = scalar_const(g, 1.0 - s.beta2, dt);
    MPSGraphTensor* eps_c = scalar_const(g, s.eps, dt);
    MPSGraphTensor* lr_c = scalar_const(g, s.lr, dt);

    MPSGraphTensor* g_sq = [g multiplicationWithPrimaryTensor:g_eff secondaryTensor:g_eff name:nil];

    MPSGraphTensor* new_m =
        [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:beta1_c
                                                        secondaryTensor:m_buf
                                                                   name:nil]
                     secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_b1
                                                        secondaryTensor:g_eff
                                                                   name:nil]
                                name:nil];
    MPSGraphTensor* new_v =
        [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:beta2_c
                                                        secondaryTensor:v_buf
                                                                   name:nil]
                     secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_b2
                                                        secondaryTensor:g_sq
                                                                   name:nil]
                                name:nil];
    MPSGraphTensor* m_hat = [g divisionWithPrimaryTensor:new_m secondaryTensor:bias1 name:nil];
    MPSGraphTensor* v_hat = [g divisionWithPrimaryTensor:new_v secondaryTensor:bias2 name:nil];
    MPSGraphTensor* v_sqrt = [g squareRootWithTensor:v_hat name:nil];
    MPSGraphTensor* denom = [g additionWithPrimaryTensor:v_sqrt secondaryTensor:eps_c name:nil];
    MPSGraphTensor* step = [g divisionWithPrimaryTensor:m_hat secondaryTensor:denom name:nil];
    if (decoupled_wd && s.weight_decay != 0.0) {
        MPSGraphTensor* wd = scalar_const(g, s.weight_decay, dt);
        step = [g additionWithPrimaryTensor:step
                            secondaryTensor:[g multiplicationWithPrimaryTensor:wd
                                                               secondaryTensor:param
                                                                          name:nil]
                                       name:nil];
    }
    MPSGraphTensor* new_p = [g subtractionWithPrimaryTensor:param
                                            secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                                               secondaryTensor:step
                                                                                          name:nil]
                                                       name:nil];
    return {new_p, new_m, new_v};
}

}  // namespace

CompiledExecutable* MpsBuilder::compile_fused_training_step(
    TensorId loss_id,
    const std::vector<TensorId>& param_ids,
    const OptimizerSpec& opt_spec,
    const std::vector<std::vector<TensorId>>& state_buf_ids_per_param,
    const std::vector<TensorId>& scalar_input_ids) {
    auto& graph = graph_;
    const auto& external_feeds = external_feeds_;
    auto* error_msg = error_msg_;

    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_fused_training_step: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_fused_training_step: param_ids must be non-empty");
    if (state_buf_ids_per_param.size() != param_ids.size())
        return fail("compile_fused_training_step: state_buf_ids size mismatch");

    // Device + emitter precheck — mirrors compile_trace_with_backward.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    }
    if (device != Device::GPU)
        return fail("compile_fused_training_step: only Device::GPU traces "
                    "are supported");
    if (const std::string empty = first_empty_tensor(graph, external_feeds); !empty.empty())
        return fail("compile_fused_training_step: zero-size tensor at " + empty +
                    " — MPSGraph aborts on one");
    if (const std::string bad_op = first_invalid_graph_op(graph, device, false, true);
        !bad_op.empty())
        return fail("compile_fused_training_step: " + bad_op);

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Ordered feed ids — same scheme as the other compile paths.
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_fused_training_step: feed " + std::to_string(tid) +
                            " is null");
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = mps_dtype_or_throw(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_fused_training_step: ") + e.what());
            }
            MPSGraphTensor* ph = [graph_obj
                placeholderWithShape:ns_shape
                            dataType:ns_dt
                                name:[NSString stringWithFormat:@"feed_%lld", (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Emit the forward + loss.
        for (const auto& node : graph.ops) {
            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty())
                    continue;  // dead-code header
                return fail("compile_fused_training_step: emitter vanished "
                            "for op '" +
                            node.name + "'");
            }
            if (const std::string why = emit_node(*emitter, ctx, node); !why.empty())
                return fail("compile_fused_training_step: " + why);
            // Emitters bind their own outputs via ctx.bind() — no auto-bind here.
        }

        // Resolve the loss tensor and param placeholders.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_fused_training_step: loss id has no "
                        "MPSGraph binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        std::vector<Shape> param_shapes;
        std::vector<Dtype> param_dtypes;
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_fused_training_step: param id " + std::to_string(pid) +
                            " has no placeholder");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
            const auto& p_impl = external_feeds.at(pid);
            param_shapes.push_back(p_impl->shape());
            param_dtypes.push_back(p_impl->dtype());
        }

        // Auto-derive gradients.  Manual VJP opt-in via env var; fall
        // through to MPSGraph autograd on coverage gap (or hard-fail
        // under LUCID_MANUAL_VJP_REQUIRE=1).
        std::vector<MPSGraphTensor*> param_grads(param_ids.size(), nil);
        bool grads_done = false;
        std::string vjp_err;
        {
            std::vector<void*> grads_void;
            switch (try_manual_vjp_grads((__bridge void*)graph_obj, ctx, graph, loss_id, param_ids,
                                         grads_void, &vjp_err)) {
            case ManualVjpStatus::Success:
                for (std::size_t i = 0; i < param_ids.size(); ++i)
                    param_grads[i] = (__bridge MPSGraphTensor*)grads_void[i];
                grads_done = true;
                break;
            case ManualVjpStatus::HardFail:
                return fail("compile_fused_training_step: "
                            "LUCID_MANUAL_VJP_REQUIRE=1 but manual VJP gap — " +
                            vjp_err);
            case ManualVjpStatus::FellBack:
            case ManualVjpStatus::Disabled:
                break;
            }
        }
        if (!grads_done) {
            if (const std::string bad = first_op_unsafe_for_autodiff(graph); !bad.empty())
                return fail("compile_fused_training_step: " + vjp_err + "; op '" + bad +
                            "' rules out MPSGraph's autodiff as the fallback");
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                [graph_obj gradientForPrimaryTensor:loss_t
                                        withTensors:param_arr
                                               name:@"lucid_fused_grads"];
            if (grad_map == nil)
                return fail("compile_fused_training_step: gradientForPrimaryTensor "
                            "returned nil");
            for (std::size_t i = 0; i < param_ids.size(); ++i)
                param_grads[i] = grad_map[param_arr[i]];
        }

        // Resolve scalar inputs (bias1, bias2 for Adam).
        std::vector<MPSGraphTensor*> scalar_tensors;
        scalar_tensors.reserve(scalar_input_ids.size());
        for (TensorId sid : scalar_input_ids) {
            void* s_void = ctx.resolve(sid);
            if (s_void == nullptr)
                return fail("compile_fused_training_step: scalar id " + std::to_string(sid) +
                            " has no placeholder");
            scalar_tensors.push_back((__bridge MPSGraphTensor*)s_void);
        }

        // Emit the optimizer update per param.  Collect (new_param,
        // new_state[]) tensors in flat output order.
        NSMutableArray<MPSGraphTensor*>* new_param_tensors =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        std::vector<std::vector<MPSGraphTensor*>> new_state_tensors;
        new_state_tensors.resize(param_ids.size());

        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            MPSGraphTensor* p = param_arr[i];
            MPSGraphTensor* grad = param_grads[i];
            if (grad == nil)
                return fail("compile_fused_training_step: no gradient for "
                            "param " +
                            std::to_string(param_ids[i]));
            MPSDataType dt = p.dataType;

            if (opt_spec.kind == OptimizerSpec::Kind::SGD) {
                MPSGraphTensor* mom = nil;
                if (opt_spec.momentum != 0.0) {
                    if (state_buf_ids_per_param[i].empty())
                        return fail("compile_fused_training_step: SGD with "
                                    "momentum requires a momentum state buffer");
                    void* m_void = ctx.resolve(state_buf_ids_per_param[i][0]);
                    if (m_void == nullptr)
                        return fail("compile_fused_training_step: momentum "
                                    "buffer placeholder missing");
                    mom = (__bridge MPSGraphTensor*)m_void;
                }
                SgdOutputs out = emit_sgd_update(graph_obj, p, grad, mom, opt_spec, dt);
                [new_param_tensors addObject:out.new_param];
                if (out.new_mom != nil)
                    new_state_tensors[i].push_back(out.new_mom);
            } else {
                // Adam / AdamW.
                if (state_buf_ids_per_param[i].size() != 2)
                    return fail("compile_fused_training_step: Adam family "
                                "requires m,v state buffers (2 per param)");
                if (scalar_tensors.size() != 2)
                    return fail("compile_fused_training_step: Adam family "
                                "requires bias1,bias2 scalar feeds");
                void* m_void = ctx.resolve(state_buf_ids_per_param[i][0]);
                void* v_void = ctx.resolve(state_buf_ids_per_param[i][1]);
                if (m_void == nullptr || v_void == nullptr)
                    return fail("compile_fused_training_step: Adam state "
                                "buffer placeholders missing");
                MPSGraphTensor* m_buf = (__bridge MPSGraphTensor*)m_void;
                MPSGraphTensor* v_buf = (__bridge MPSGraphTensor*)v_void;
                bool decoupled = (opt_spec.kind == OptimizerSpec::Kind::ADAMW);
                AdamOutputs out =
                    emit_adam_update(graph_obj, p, grad, m_buf, v_buf, scalar_tensors[0],
                                     scalar_tensors[1], opt_spec, dt, decoupled);
                [new_param_tensors addObject:out.new_param];
                new_state_tensors[i].push_back(out.new_m);
                new_state_tensors[i].push_back(out.new_v);
            }
        }

        // Assemble target list in declared output order:
        // [loss, new_p_0, ..., new_p_N-1, new_state[0][0], new_state[0][1], ...]
        NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray array];
        [target_arr addObject:loss_t];
        for (NSUInteger i = 0; i < [new_param_tensors count]; ++i)
            [target_arr addObject:new_param_tensors[i]];
        for (std::size_t i = 0; i < new_state_tensors.size(); ++i)
            for (MPSGraphTensor* t : new_state_tensors[i])
                [target_arr addObject:t];

        // Build feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = mps_dtype_or_throw(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] = [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                                          dataType:ns_dt];
        }

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled =
            [graph_obj compileWithDevice:mps_device
                                   feeds:feed_dict
                           targetTensors:target_arr
                        targetOperations:nil
                   compilationDescriptor:make_compile_descriptor(carries_half_precision(graph))];
        if (compiled == nil)
            return fail("compile_fused_training_step: MPSGraph compile "
                        "returned nil");

        // Mint trace ids for the new_param / new_state outputs (so the
        // CompiledExecutable's id list is unambiguous).  Start past
        // the max id we've seen — the Python wrapper may have
        // injected state-buffer / scalar feeds with ids past
        // ``graph.next_id`` (which is itself a read-only property on
        // the Python side, so the wrapper can't update it), so we
        // can't trust ``graph.next_id + 1`` alone.
        TensorId max_id = graph.next_id;
        for (const auto& [tid, _] : external_feeds)
            if (tid > max_id)
                max_id = tid;
        std::vector<TensorId> aux_output_ids;
        TensorId next_id = max_id + 1;
        for (std::size_t i = 0; i < param_ids.size(); ++i)
            aux_output_ids.push_back(next_id++);
        for (std::size_t i = 0; i < new_state_tensors.size(); ++i)
            for (std::size_t k = 0; k < new_state_tensors[i].size(); ++k)
                aux_output_ids.push_back(next_id++);

        // Find loss meta in the trace.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_fused_training_step: loss meta missing");

        // Build the per-output shape/dtype list parallel to target_arr.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);
        // new_param_i: shape == param_shapes[i].
        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            output_shapes.push_back(param_shapes[i]);
            output_dtypes.push_back(param_dtypes[i]);
        }
        // new_state[i][k]: shape == state buffer's trace impl shape.
        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            for (std::size_t k = 0; k < state_buf_ids_per_param[i].size(); ++k) {
                TensorId sb = state_buf_ids_per_param[i][k];
                const auto& sb_impl = external_feeds.at(sb);
                output_shapes.push_back(sb_impl->shape());
                output_dtypes.push_back(sb_impl->dtype());
            }
        }

        auto exe = std::make_unique<CompiledExecutable>();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe.release();
    }
}

// ── Generic fused step (Phase 1.8) ──────────────────────────────────

CompiledExecutable*
MpsBuilder::compile_generic_fused_step(TensorId loss_id,
                                       const std::vector<TensorId>& param_ids,
                                       const std::vector<TensorId>& ghost_grad_ids,
                                       const std::vector<TensorId>& output_target_ids) {
    auto& graph = graph_;
    const auto& external_feeds = external_feeds_;
    auto* error_msg = error_msg_;

    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_generic_fused_step: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_generic_fused_step: param_ids must be non-empty");
    if (ghost_grad_ids.size() != param_ids.size())
        return fail("compile_generic_fused_step: ghost_grad_ids size != param_ids size");
    if (const TensorId off = first_param_off_loss_path(graph, loss_id, param_ids); off >= 0)
        return fail("compile_generic_fused_step: param id " + std::to_string(off) +
                    " does not reach the loss — MPSGraph aborts the process when asked for "
                    "its gradient; leave it out of param_ids");

    const std::unordered_set<TensorId> ghost_set(ghost_grad_ids.begin(), ghost_grad_ids.end());

    // Device + emitter precheck.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    }
    if (device != Device::GPU)
        return fail("compile_generic_fused_step: only Device::GPU traces supported");
    if (const std::string empty = first_empty_tensor(graph, external_feeds); !empty.empty())
        return fail("compile_generic_fused_step: zero-size tensor at " + empty +
                    " — MPSGraph aborts on one");
    if (const std::string bad_op = first_invalid_graph_op(graph, device, false, true);
        !bad_op.empty())
        return fail("compile_generic_fused_step: " + bad_op);

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Build placeholders for every external_feed EXCEPT the ghost
        // grads (those are bound to derived gradients after the forward).
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            if (ghost_set.find(tid) == ghost_set.end())
                ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_generic_fused_step: null feed " + std::to_string(tid));
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = mps_dtype_or_throw(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_generic_fused_step: ") + e.what());
            }
            MPSGraphTensor* ph = [graph_obj
                placeholderWithShape:ns_shape
                            dataType:ns_dt
                                name:[NSString stringWithFormat:@"feed_%lld", (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Build the param-tensor array (we'll need it for the grad call).
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_generic_fused_step: param id " + std::to_string(pid) +
                            " has no placeholder (not in external_feeds?)");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Pre-compute the trace-wide "consumed by some op" set so the
        // multi-output emitters (split / split_at / unbind / chunk /
        // topk / lstm) bind every piece that any downstream op reads.
        // Without this set, ``ctx.is_consumed`` returns false for every
        // piece and the emitters skip binding piece[1+] — embedding /
        // attention chains that consume piece[1+] then fail with
        // ``emitter 'embedding' returned nullptr`` because the
        // indices id has no MPSGraph binding.
        {
            std::unordered_set<TensorId> trace_consumed;
            for (const auto& n : graph.ops)
                for (TensorId iid : n.inputs)
                    if (iid >= 0)
                        trace_consumed.insert(iid);
            // Also mark the explicit graph targets as consumed (loss +
            // output_target_ids) so emitters that produce them as multi-
            // output slots still bind them.
            trace_consumed.insert(loss_id);
            for (TensorId tid : output_target_ids)
                if (tid >= 0)
                    trace_consumed.insert(tid);
            mark_vjp_regions_consumed(graph, trace_consumed);
            ctx.set_consumed_inputs(trace_consumed);
            ctx.set_softmax_outputs(collect_softmax_outputs(graph));
        }

        // Emit ops in trace order.  Before reaching the first op that
        // consumes a ghost grad, derive grads via
        // ``gradientForPrimaryTensor:withTensors:`` and bind each
        // ghost_grad_id to the derived gradient tensor for the
        // corresponding param.
        bool grads_derived = false;

        auto derive_grads_now = [&]() -> bool {
            if (grads_derived)
                return true;
            void* loss_void = ctx.resolve(loss_id);
            if (loss_void == nullptr) {
                if (error_msg)
                    *error_msg = "compile_generic_fused_step: loss id has no "
                                 "MPSGraph binding when reaching opt phase";
                return false;
            }
            MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

            // Manual VJP (LUCID_MANUAL_VJP=1): walk the trace in reverse
            // and emit per-op backward subgraphs ourselves.  On VJP
            // coverage gap we fall through to MPSGraph autograd, unless
            // LUCID_MANUAL_VJP_REQUIRE=1.
            std::string vjp_err;
            {
                std::vector<void*> grads;
                switch (try_manual_vjp_grads((__bridge void*)graph_obj, ctx, graph, loss_id,
                                             param_ids, grads, &vjp_err)) {
                case ManualVjpStatus::Success:
                    for (std::size_t i = 0; i < param_ids.size(); ++i)
                        ctx.bind(ghost_grad_ids[i], grads[i]);
                    grads_derived = true;
                    return true;
                case ManualVjpStatus::HardFail:
                    if (error_msg)
                        *error_msg = "compile_generic_fused_step: "
                                     "LUCID_MANUAL_VJP_REQUIRE=1 but manual VJP gap — " +
                                     vjp_err;
                    return false;
                case ManualVjpStatus::FellBack:
                case ManualVjpStatus::Disabled:
                    break;
                }
            }

            if (const std::string bad = first_op_unsafe_for_autodiff(graph); !bad.empty()) {
                if (error_msg)
                    *error_msg = "compile_generic_fused_step: " + vjp_err + "; op '" + bad +
                                 "' rules out MPSGraph's autodiff as the fallback";
                return false;
            }
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                [graph_obj gradientForPrimaryTensor:loss_t
                                        withTensors:param_arr
                                               name:@"lucid_generic_grads"];
            if (grad_map == nil) {
                if (error_msg)
                    *error_msg = "compile_generic_fused_step: gradientForPrimaryTensor "
                                 "returned nil";
                return false;
            }
            for (std::size_t i = 0; i < param_ids.size(); ++i) {
                MPSGraphTensor* g_t = grad_map[param_arr[i]];
                if (g_t == nil) {
                    if (error_msg)
                        *error_msg = "compile_generic_fused_step: no gradient for "
                                     "param " +
                                     std::to_string(param_ids[i]);
                    return false;
                }
                ctx.bind(ghost_grad_ids[i], (__bridge void*)g_t);
            }
            grads_derived = true;
            return true;
        };

        for (const auto& node : graph.ops) {
            // Detect entry into the opt-update phase: any input is a
            // ghost grad id.
            bool uses_ghost = false;
            for (TensorId iid : node.inputs) {
                if (ghost_set.find(iid) != ghost_set.end()) {
                    uses_ghost = true;
                    break;
                }
            }
            if (uses_ghost && !grads_derived) {
                if (!derive_grads_now())
                    return nullptr;
            }

            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty() && result_is_a_host_constant(node.name))
                    continue;  // host constant — see compile_trace
                return fail("compile_generic_fused_step: emitter vanished "
                            "for op '" +
                            node.name + "'");
            }
            if (const std::string why = emit_node(*emitter, ctx, node); !why.empty())
                return fail("compile_generic_fused_step: " + why);
            // Emitters bind their own outputs via ctx.bind() — no auto-bind here.
        }

        // If the trace doesn't actually have any op that consumes a
        // ghost grad (degenerate case — opt is a no-op?), derive grads
        // anyway so the output_target_ids can resolve gradient-only
        // outputs.  Cheap when grad map isn't consumed.
        if (!grads_derived && !ghost_grad_ids.empty()) {
            if (!derive_grads_now())
                return nullptr;
        }

        // Resolve targets: loss + output_target_ids.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_generic_fused_step: loss id has no "
                        "MPSGraph binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray array];
        [target_arr addObject:loss_t];
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;

        // Loss meta.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_generic_fused_step: loss meta missing");
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);

        // Per-output_target_id meta.  Search graph.ops for matching
        // TensorMeta.
        std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
        for (const auto& node : graph.ops)
            for (const auto& meta : node.outputs)
                id_to_meta[meta.id] = {meta.shape, meta.dtype};

        for (TensorId tid : output_target_ids) {
            void* t_void = ctx.resolve(tid);
            if (t_void == nullptr)
                return fail("compile_generic_fused_step: output target id " + std::to_string(tid) +
                            " has no MPSGraph binding");
            [target_arr addObject:(__bridge MPSGraphTensor*)t_void];
            auto it = id_to_meta.find(tid);
            if (it == id_to_meta.end()) {
                // Maybe it's an external feed (shouldn't happen for opt
                // outputs but handle defensively).
                auto fit = external_feeds.find(tid);
                if (fit == external_feeds.end() || !fit->second)
                    return fail("compile_generic_fused_step: output target "
                                "id " +
                                std::to_string(tid) + " not in trace");
                output_shapes.push_back(fit->second->shape());
                output_dtypes.push_back(fit->second->dtype());
            } else {
                output_shapes.push_back(it->second.first);
                output_dtypes.push_back(it->second.second);
            }
        }

        // Feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = mps_dtype_or_throw(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] = [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                                          dataType:ns_dt];
        }

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled =
            [graph_obj compileWithDevice:mps_device
                                   feeds:feed_dict
                           targetTensors:target_arr
                        targetOperations:nil
                   compilationDescriptor:make_compile_descriptor(carries_half_precision(graph))];
        if (compiled == nil)
            return fail("compile_generic_fused_step: MPSGraph compile returned nil");

        // Mint aux ids past the max observed id (output_target_ids are
        // already in the trace, so the executable's grad_output_ids
        // can carry them directly).
        std::vector<TensorId> aux_output_ids(output_target_ids);

        auto exe = std::make_unique<CompiledExecutable>();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe.release();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Stateful-variables variant of compile_generic_fused_step.
//
// See the header doc for the full design rationale.  Implementation
// mirrors ``compile_generic_fused_step`` line-by-line but with three
// surgical insertions:
//
//   1. For every feed in ``variable_pairs``, the feed becomes
//      ``variableWithData:`` (initialized from the Lucid Tensor's
//      current MTLBuffer contents) rather than ``placeholderWithShape:``.
//      The feed slot is removed from the executable's ``input_ids``.
//   2. After the trace emits the new value for each variable, the
//      builder issues ``assignVariable:`` and adds the operation to
//      the ``targetOperations`` array so it executes on every call.
//      A ``readVariable:`` op is also created and added to
//      ``targetTensors`` in place of the original write_id — so
//      ``run_executable_inplace`` can bind the Lucid Tensor's
//      existing MTLBuffer and receive the post-assign value with
//      no buffer alloc.
//   3. The original write_id entry in ``output_target_ids`` is
//      replaced by the readVariable's trace id in
//      ``grad_output_ids`` (same slot index — caller is unaffected).
// ─────────────────────────────────────────────────────────────────────
CompiledExecutable* MpsBuilder::compile_generic_fused_step_with_vars(
    TensorId loss_id,
    const std::vector<TensorId>& param_ids,
    const std::vector<TensorId>& ghost_grad_ids,
    const std::vector<TensorId>& output_target_ids,
    const std::vector<std::pair<TensorId, TensorId>>& variable_pairs) {
    auto& graph = graph_;
    const auto& external_feeds = external_feeds_;
    auto* error_msg = error_msg_;

    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_generic_fused_step_with_vars: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_generic_fused_step_with_vars: param_ids must be non-empty");
    if (ghost_grad_ids.size() != param_ids.size())
        return fail("compile_generic_fused_step_with_vars: ghost_grad_ids size != param_ids size");

    // If no variables requested, defer to the non-vars path on this
    // same MpsBuilder instance — no need to reconstruct.
    if (variable_pairs.empty()) {
        return this->compile_generic_fused_step(loss_id, param_ids, ghost_grad_ids,
                                                output_target_ids);
    }
    if (const TensorId off = first_param_off_loss_path(graph, loss_id, param_ids); off >= 0)
        return fail("compile_generic_fused_step_with_vars: param id " + std::to_string(off) +
                    " does not reach the loss — MPSGraph aborts the process when asked for "
                    "its gradient; leave it out of param_ids");

    const std::unordered_set<TensorId> ghost_set(ghost_grad_ids.begin(), ghost_grad_ids.end());

    // Set of feed ids that are promoted to variables, and a map from
    // each var-feed-id to its corresponding write_id (the new value
    // produced by the trace that gets assigned back into the variable).
    std::unordered_set<TensorId> var_feed_set;
    std::unordered_map<TensorId, TensorId> feed_to_write;
    std::unordered_set<TensorId> var_write_set;
    for (const auto& [f, w] : variable_pairs) {
        if (external_feeds.find(f) == external_feeds.end())
            return fail("compile_generic_fused_step_with_vars: variable feed id " +
                        std::to_string(f) + " not in external_feeds");
        var_feed_set.insert(f);
        feed_to_write[f] = w;
        var_write_set.insert(w);
    }

    Device device = Device::CPU;
    for (const auto& op : graph.ops)
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    if (device != Device::GPU)
        return fail("compile_generic_fused_step_with_vars: only GPU supported");
    if (const std::string empty = first_empty_tensor(graph, external_feeds); !empty.empty())
        return fail("compile_generic_fused_step_with_vars: zero-size tensor at " + empty +
                    " — MPSGraph aborts on one");
    if (const std::string bad_op = first_invalid_graph_op(graph, device, false, true);
        !bad_op.empty())
        return fail("compile_generic_fused_step_with_vars: " + bad_op);

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Variables init reads raw MTLBuffer contents at compile time
        // — flush pending MLX async work first so the snapshot reflects
        // the current parameter values rather than half-evaluated lazy
        // state.  Without this the variableWithData: source can be
        // mid-stream and the resulting variable holds garbage (or the
        // contents read blocks indefinitely).
        lucid::gpu::mps::wait_all();

        // Step 1: build placeholders for non-var, non-ghost feeds AND
        // variables for var_feed_set.  ``ordered_feed_ids`` records only
        // the placeholder feeds so ``input_ids`` excludes variables.
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        std::vector<TensorId> sorted_all_feeds;
        sorted_all_feeds.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            if (ghost_set.find(tid) == ghost_set.end())
                sorted_all_feeds.push_back(tid);
        std::sort(sorted_all_feeds.begin(), sorted_all_feeds.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;

        for (TensorId tid : sorted_all_feeds) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_generic_fused_step_with_vars: null feed " +
                            std::to_string(tid));
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = mps_dtype_or_throw(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_generic_fused_step_with_vars: ") + e.what());
            }

            if (var_feed_set.find(tid) != var_feed_set.end()) {
                // Variable: snapshot the current MTLBuffer contents
                // into NSData for ``variableWithData:``.  The data is
                // copied into MPSGraph's own internal storage; we can
                // safely free the NSData after.
                if (impl->device() != Device::GPU)
                    return fail("compile_generic_fused_step_with_vars: variable feed not on GPU");
                const auto& gs = std::get<GpuStorage>(impl->storage());
                if (!gs.arr)
                    return fail(
                        "compile_generic_fused_step_with_vars: variable feed has no MLX array");
                lucid::gpu::mps::BufferView v = lucid::gpu::mps::array_to_buffer(*gs.arr);
                id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)v.mtl_buffer;
                // Inline shape_nbytes — same formula as
                // CompiledExecutable.mm's detail::shape_nbytes (anon
                // namespace there is unreachable from this TU).  Both
                // ask ``dtype_size``; the two-case guess they used to
                // share read every eight-byte dtype as four.
                std::size_t nelem = 1;
                for (std::int64_t d : feed_shape)
                    nelem *= static_cast<std::size_t>(d);
                const std::size_t nbytes = nelem * dtype_size(feed_dtype);
                NSData* init_data = [NSData dataWithBytes:[src_buf contents] length:nbytes];
                MPSGraphTensor* var = [graph_obj
                    variableWithData:init_data
                               shape:ns_shape
                            dataType:ns_dt
                                name:[NSString stringWithFormat:@"var_%lld", (long long)tid]];
                ctx.bind(tid, (__bridge void*)var);
                // NOTE: var feeds NOT added to feed_tensors / input_shapes.
            } else {
                MPSGraphTensor* ph = [graph_obj
                    placeholderWithShape:ns_shape
                                dataType:ns_dt
                                    name:[NSString stringWithFormat:@"feed_%lld", (long long)tid]];
                ctx.bind(tid, (__bridge void*)ph);
                feed_tensors.push_back(ph);
                input_shapes.push_back(feed_shape);
                input_dtypes.push_back(feed_dtype);
                ordered_feed_ids.push_back(tid);
            }
        }

        // Step 2: param array for gradient derivation.  Param feeds
        // are now variables — ctx.resolve returns the variable tensor.
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_generic_fused_step_with_vars: param id " +
                            std::to_string(pid) + " has no binding");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Pre-compute trace-wide consumed set — see parallel block in
        // compile_generic_fused_step.  Required for multi-output emitters
        // (split / split_at / unbind / chunk / topk / lstm) to bind every
        // piece any downstream op reads.
        {
            std::unordered_set<TensorId> trace_consumed;
            for (const auto& n : graph.ops)
                for (TensorId iid : n.inputs)
                    if (iid >= 0)
                        trace_consumed.insert(iid);
            trace_consumed.insert(loss_id);
            for (TensorId tid : output_target_ids)
                if (tid >= 0)
                    trace_consumed.insert(tid);
            mark_vjp_regions_consumed(graph, trace_consumed);
            ctx.set_consumed_inputs(trace_consumed);
            ctx.set_softmax_outputs(collect_softmax_outputs(graph));
        }

        // Step 3: emit trace ops with ghost-grad derivation, same as
        // the non-variables path.
        bool grads_derived = false;
        auto derive_grads_now = [&]() -> bool {
            if (grads_derived)
                return true;
            void* loss_void = ctx.resolve(loss_id);
            if (loss_void == nullptr) {
                if (error_msg)
                    *error_msg = "compile_generic_fused_step_with_vars: loss id has no binding";
                return false;
            }
            MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

            // Manual VJP path (LUCID_MANUAL_VJP=1) — mirror of the
            // non-variables fused-step site above.
            std::string vjp_err;
            {
                std::vector<void*> grads;
                switch (try_manual_vjp_grads((__bridge void*)graph_obj, ctx, graph, loss_id,
                                             param_ids, grads, &vjp_err)) {
                case ManualVjpStatus::Success:
                    for (std::size_t i = 0; i < param_ids.size(); ++i)
                        ctx.bind(ghost_grad_ids[i], grads[i]);
                    grads_derived = true;
                    return true;
                case ManualVjpStatus::HardFail:
                    if (error_msg)
                        *error_msg = "compile_generic_fused_step_with_vars: "
                                     "LUCID_MANUAL_VJP_REQUIRE=1 but manual VJP gap — " +
                                     vjp_err;
                    return false;
                case ManualVjpStatus::FellBack:
                case ManualVjpStatus::Disabled:
                    break;
                }
            }

            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                [graph_obj gradientForPrimaryTensor:loss_t
                                        withTensors:param_arr
                                               name:@"lucid_var_grads"];
            if (grad_map == nil) {
                if (error_msg)
                    *error_msg =
                        "compile_generic_fused_step_with_vars: gradientForPrimaryTensor nil";
                return false;
            }
            for (std::size_t i = 0; i < param_ids.size(); ++i) {
                MPSGraphTensor* g_t = grad_map[param_arr[i]];
                if (g_t == nil) {
                    if (error_msg)
                        *error_msg =
                            "compile_generic_fused_step_with_vars: no gradient for param " +
                            std::to_string(param_ids[i]);
                    return false;
                }
                ctx.bind(ghost_grad_ids[i], (__bridge void*)g_t);
            }
            grads_derived = true;
            return true;
        };

        for (const auto& node : graph.ops) {
            bool uses_ghost = false;
            for (TensorId iid : node.inputs)
                if (ghost_set.find(iid) != ghost_set.end()) {
                    uses_ghost = true;
                    break;
                }
            if (uses_ghost && !grads_derived)
                if (!derive_grads_now())
                    return nullptr;

            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty())
                    continue;
                return fail("compile_generic_fused_step_with_vars: emitter vanished for op '" +
                            node.name + "'");
            }
            if (const std::string why = emit_node(*emitter, ctx, node); !why.empty())
                return fail("compile_generic_fused_step_with_vars: " + why);
            // Emitters bind their own outputs via ctx.bind() — no auto-bind here.
        }
        if (!grads_derived && !ghost_grad_ids.empty()) {
            if (!derive_grads_now())
                return nullptr;
        }

        // Step 4: emit assignVariable for each (feed, write) pair +
        // readVariable into a target tensor we'll expose as an output
        // (so per-call the new value flushes into the Lucid Tensor's
        // existing MTLBuffer with no fresh allocation).
        NSMutableArray<MPSGraphOperation*>* target_ops_arr = [NSMutableArray array];

        // Loss target.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_generic_fused_step_with_vars: loss has no binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
        NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray array];
        [target_arr addObject:loss_t];

        // Loss meta.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops)
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        if (!loss_found)
            return fail("compile_generic_fused_step_with_vars: loss meta missing");
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);

        // Build id_to_meta map once for output_target_ids lookups.
        std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
        for (const auto& node : graph.ops)
            for (const auto& meta : node.outputs)
                id_to_meta[meta.id] = {meta.shape, meta.dtype};

        // Process each output_target_id.  If it's a var write, emit
        // assignVariable + readVariable; otherwise add directly as a
        // target tensor.
        std::vector<TensorId> aux_output_ids;
        aux_output_ids.reserve(output_target_ids.size());
        for (TensorId tid : output_target_ids) {
            void* new_value_void = ctx.resolve(tid);
            if (new_value_void == nullptr)
                return fail("compile_generic_fused_step_with_vars: output target id " +
                            std::to_string(tid) + " has no binding");
            MPSGraphTensor* new_value_t = (__bridge MPSGraphTensor*)new_value_void;

            // Is this id a "write" for any variable?
            TensorId var_feed_for_write = -1;
            for (const auto& [f, w] : feed_to_write)
                if (w == tid) {
                    var_feed_for_write = f;
                    break;
                }

            if (var_feed_for_write >= 0) {
                // Variable write: emit assignVariable as a SIDE
                // EFFECT (added to targetOperations) and use the
                // computed ``new_value_t`` directly as the output
                // tensor.  This mirrors the working standalone
                // pattern in /tmp/var_grad.mm — using
                // ``readVariable:`` after ``assignVariable:`` inside
                // the same compile unit creates a read-after-write
                // dependency on the same variable, which interacts
                // poorly with MPSGraph's autograd scheduler (the
                // backward pass already reads the variable, so an
                // additional read introduces extra scheduling
                // constraints that we empirically observed cause a
                // hang on macOS 26 SDK).  ``new_value_t`` and
                // ``readVariable`` post-assign are bit-identical —
                // assignVariable copies new_value into the variable
                // verbatim — so we lose nothing semantically.
                void* var_void = ctx.resolve(var_feed_for_write);
                MPSGraphTensor* var_t = (__bridge MPSGraphTensor*)var_void;
                MPSGraphOperation* assign_op = [graph_obj
                       assignVariable:var_t
                    withValueOfTensor:new_value_t
                                 name:[NSString stringWithFormat:@"assign_%lld", (long long)tid]];
                [target_ops_arr addObject:assign_op];
                [target_arr addObject:new_value_t];
            } else {
                // Plain output target: add to targetTensors directly.
                [target_arr addObject:new_value_t];
            }

            auto it = id_to_meta.find(tid);
            if (it == id_to_meta.end()) {
                auto fit = external_feeds.find(tid);
                if (fit == external_feeds.end() || !fit->second)
                    return fail("compile_generic_fused_step_with_vars: output target id " +
                                std::to_string(tid) + " not in trace");
                output_shapes.push_back(fit->second->shape());
                output_dtypes.push_back(fit->second->dtype());
            } else {
                output_shapes.push_back(it->second.first);
                output_dtypes.push_back(it->second.second);
            }
            aux_output_ids.push_back(tid);
        }

        // Feed dict only includes non-variable feeds.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = mps_dtype_or_throw(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] = [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                                          dataType:ns_dt];
        }

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled =
            [graph_obj compileWithDevice:mps_device
                                   feeds:feed_dict
                           targetTensors:target_arr
                        targetOperations:[target_ops_arr count] > 0 ? target_ops_arr : nil
                   compilationDescriptor:make_compile_descriptor(carries_half_precision(graph))];
        if (compiled == nil)
            return fail("compile_generic_fused_step_with_vars: MPSGraph compile returned nil");

        auto exe = std::make_unique<CompiledExecutable>();
        exe->executable = compiled;
        // Retain the source MPSGraph so its variable storage outlives
        // the compile autoreleasepool.  Without this, MPSGraph releases
        // the variable's backing MTLBuffer along with the graph object;
        // subsequent ``runWithMTLCommandQueue:`` calls segfault inside
        // ``GPU::VarHandleOpHandler::encodeOp`` when the executor tries
        // to dereference the dangling variable handle.  The retain is
        // released in ``~CompiledExecutable``.
        exe->source_graph = (__bridge_retained void*)graph_obj;
        exe->input_ids = std::move(ordered_feed_ids);
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe.release();
    }
}

// ────────────────────────────────────────────────────────────────────
// Free-function forwarders — preserve the pre-class API.
// Every existing caller (Python bindings, ExecutableCache, tests)
// imports these by name; they construct a transient :class:`MpsBuilder`
// and delegate to the matching method.  See MpsBuilder.h for the
// class design rationale.
// ────────────────────────────────────────────────────────────────────

CompiledExecutable* compile_trace(const TraceGraph& graph,
                                  const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                                  std::string* error_msg,
                                  bool dynamic_batch,
                                  const std::vector<TensorId>& param_ids,
                                  const std::vector<TensorId>& explicit_outputs) {
    return MpsBuilder(graph, external_feeds, error_msg)
        .compile_trace(dynamic_batch, param_ids, explicit_outputs);
}

CompiledExecutable*
compile_trace_with_backward(const TraceGraph& graph,
                            const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                            TensorId loss_id,
                            const std::vector<TensorId>& param_ids,
                            std::string* error_msg,
                            bool dynamic_batch,
                            const std::vector<TensorId>& extra_output_ids) {
    return MpsBuilder(graph, external_feeds, error_msg)
        .compile_trace_with_backward(loss_id, param_ids, dynamic_batch, extra_output_ids);
}

CompiledExecutable*
compile_fused_training_step(const TraceGraph& graph,
                            const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                            TensorId loss_id,
                            const std::vector<TensorId>& param_ids,
                            const OptimizerSpec& opt_spec,
                            const std::vector<std::vector<TensorId>>& state_buf_ids_per_param,
                            const std::vector<TensorId>& scalar_input_ids,
                            std::string* error_msg) {
    return MpsBuilder(graph, external_feeds, error_msg)
        .compile_fused_training_step(loss_id, param_ids, opt_spec, state_buf_ids_per_param,
                                     scalar_input_ids);
}

CompiledExecutable*
compile_generic_fused_step(const TraceGraph& graph,
                           const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                           TensorId loss_id,
                           const std::vector<TensorId>& param_ids,
                           const std::vector<TensorId>& ghost_grad_ids,
                           const std::vector<TensorId>& output_target_ids,
                           std::string* error_msg) {
    return MpsBuilder(graph, external_feeds, error_msg)
        .compile_generic_fused_step(loss_id, param_ids, ghost_grad_ids, output_target_ids);
}

CompiledExecutable* compile_generic_fused_step_with_vars(
    const TraceGraph& graph,
    const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
    TensorId loss_id,
    const std::vector<TensorId>& param_ids,
    const std::vector<TensorId>& ghost_grad_ids,
    const std::vector<TensorId>& output_target_ids,
    const std::vector<std::pair<TensorId, TensorId>>& variable_pairs,
    std::string* error_msg) {
    return MpsBuilder(graph, external_feeds, error_msg)
        .compile_generic_fused_step_with_vars(loss_id, param_ids, ghost_grad_ids, output_target_ids,
                                              variable_pairs);
}

}  // namespace lucid::compile
