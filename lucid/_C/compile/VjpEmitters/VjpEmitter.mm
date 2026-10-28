// lucid/_C/compile/VjpEmitters/VjpEmitter.mm
//
// Registry + BackwardContext + BackwardWalker implementation.
// MPSGraph-typed since :func:`BackwardContext::accumulate_grad` and
// :func:`BackwardContext::unreduce` emit MPSGraph ops directly.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "../../core/Dtype.h"
#include "../OpEmitters/_SignedZero.h"
#include "VjpEmitter.h"
#include "_VjpHelpers.h"

namespace lucid::compile {

// ────────────────────────────────────────────────────────────────────
// Registry
// ────────────────────────────────────────────────────────────────────

namespace {

struct VjpRegistry {
    std::mutex mu;
    std::unordered_map<std::string, std::unique_ptr<VjpEmitter>> map;
};

VjpRegistry& vjp_registry() {
    static VjpRegistry* instance = new VjpRegistry();
    return *instance;
}

// Trace ops that have no meaningful gradient (factories, integer-only
// shape ops, eager-fallback stubs).  The walker treats them as
// grad-sinks: it does not require a VJP emitter for them and does not
// propagate grads through their inputs.
const std::unordered_set<std::string>& no_grad_ops() {
    static const std::unordered_set<std::string> s{
        // factories — no inputs, just produce constants
        "zeros",
        "ones",
        "full",
        "arange",
        "linspace",
        "eye",
        "empty",
        "randn",
        "rand",
        "randint",
        "uniform",
        "normal",
        // shape-only / integer ops that don't need gradients
        "cast_i8",
        "cast_i16",
        "cast_i32",
        "cast_i64",
        "cast_u8",
        "cast_b8",
        // boolean comparison ops — produce bool tensors that are
        // never on the backward differentiation path; treat as
        // grad-sinks so the walker doesn't error out looking for a
        // VJP that wouldn't make sense anyway.
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "isnan",
        "isinf",
        "isfinite",
        // argmax / argmin produce integer indices — also grad-sinks.
        "argmax",
        "argmin",
        // detach — the value, cut from its input's gradient.  A sink
        // rather than a no-op VJP so fused_step's reach test, which reads
        // this set through ``vjp_registration_status``, leaves a parameter
        // seen only through it out of the update, as eager leaves its grad
        // None.  MPSGraph's autodiff would differentiate straight through
        // the identity, so it stays off that fallback's safe list.
        "detach",
        // bitcast — raw IEEE-754 bit reinterpretation (used in no_grad
        // subnormal normalization/restoration on Metal); never carries a
        // gradient across the bit boundary.
        "bitcast",
    };
    return s;
}

// Whether a gradient can flow back through ``node``: some result is a
// floating or complex value.  fused_step's reach test
// (``lucid/compile/_entry/fused_step.py``) applies the same rule.
bool carries_gradient(const OpNode& node) {
    for (const auto& om : node.outputs)
        if (is_floating_point(om.dtype) || is_complex(om.dtype))
            return true;
    return false;
}

void propagate_region_dependents(const VjpRegion& region, std::unordered_set<TensorId>& depends) {
    bool dep = false;
    for (std::size_t k = 0; k < region.input_ids.size() && k < region.input_grad_ids.size(); ++k) {
        if (region.input_ids[k] >= 0 && region.input_grad_ids[k] >= 0 &&
            depends.count(region.input_ids[k]) != 0) {
            dep = true;
            break;
        }
    }
    if (!dep)
        return;
    for (const auto& om : region.output_metas) {
        if (is_floating_point(om.dtype) || is_complex(om.dtype))
            depends.insert(om.id);
    }
}

std::unordered_set<TensorId> param_dependents(const TraceGraph& trace,
                                              const std::vector<TensorId>& param_ids) {
    const auto& ops = trace.ops;
    const auto& sink = no_grad_ops();
    std::unordered_map<std::size_t, const VjpRegion*> region_by_begin;
    for (const auto& r : trace.vjp_regions) {
        if (r.fwd_end > r.fwd_begin)
            region_by_begin[r.fwd_begin] = &r;
    }
    std::unordered_set<TensorId> depends(param_ids.begin(), param_ids.end());
    std::size_t idx = 0;
    while (idx < ops.size()) {
        auto rit = region_by_begin.find(idx);
        if (rit != region_by_begin.end()) {
            propagate_region_dependents(*rit->second, depends);
            idx = rit->second->fwd_end;
            continue;
        }
        const OpNode& node = ops[idx++];
        if (sink.find(node.name) != sink.end() || !carries_gradient(node))
            continue;
        for (TensorId in : node.inputs) {
            if (in >= 0 && depends.count(in) != 0) {
                for (const auto& om : node.outputs)
                    depends.insert(om.id);
                break;
            }
        }
    }
    return depends;
}

bool bind_vjp_region_cotangents(MPSGraph* graph,
                                BackwardContext& bctx,
                                const VjpRegion& region,
                                std::string* error_msg) {
    BuilderContext& fctx = bctx.forward_ctx();
    for (std::size_t j = 0; j < region.output_metas.size() && j < region.cotangent_ids.size();
         ++j) {
        const TensorMeta& om = region.output_metas[j];
        const TensorId cid = region.cotangent_ids[j];
        if (cid < 0)
            continue;
        auto dt_opt = mps_dtype_of(om.dtype);
        if (!dt_opt) {
            if (error_msg)
                *error_msg = "manual_vjp: unsupported cotangent dtype in VjpRegion";
            return false;
        }
        void* go = bctx.resolve_grad(om.id);
        MPSGraphTensor* go_t = go != nullptr ? cast_if_needed(graph, as_tensor(go), *dt_opt)
                                             : [graph constantWithScalar:0.0
                                                                   shape:shape_to_ns(om.shape)
                                                                dataType:*dt_opt];
        fctx.bind(cid, (__bridge void*)go_t);
        if (go == nullptr)
            mark_zero_splat(fctx, cid, om.dtype, 0.0, go_t);
    }
    return true;
}

bool emit_vjp_region_ops(MPSGraph* graph,
                         BuilderContext& fctx,
                         const VjpRegion& region,
                         std::string* error_msg) {
    for (const auto& bwd_node : region.bwd_ops) {
        OpEmitter* em = find_emitter(bwd_node.name);
        if (em == nullptr) {
            if (bwd_node.inputs.empty())
                continue;
            if (error_msg)
                *error_msg = "manual_vjp: custom backward op '" + bwd_node.name + "' failed";
            return false;
        }
        if (!em->emit(fctx, bwd_node)) {
            if (error_msg)
                *error_msg = "manual_vjp: custom backward op '" + bwd_node.name + "' failed";
            return false;
        }
        propagate_zero_splat(fctx, bwd_node);
        for (const auto& meta : bwd_node.outputs) {
            void* piece = fctx.resolve(meta.id);
            auto dt_opt = mps_dtype_of(meta.dtype);
            if (piece == nullptr || !dt_opt)
                continue;
            MPSGraphTensor* pt = as_tensor(piece);
            if (pt != nil && pt.dataType != *dt_opt) {
                pt = cast_mps_tensor(graph, pt, *dt_opt, @"vjp_region_held_dtype");
                fctx.bind(meta.id, (__bridge void*)pt);
            }
        }
    }
    return true;
}

bool emit_vjp_region(MPSGraph* graph,
                     BackwardContext& bctx,
                     const VjpRegion& region,
                     const std::unordered_set<TensorId>& depends,
                     std::string* error_msg) {
    bool any_demand = false;
    for (const auto& om : region.output_metas) {
        if (depends.count(om.id) != 0 && bctx.resolve_grad(om.id) != nullptr) {
            any_demand = true;
            break;
        }
    }
    if (!any_demand)
        return true;
    if (!bind_vjp_region_cotangents(graph, bctx, region, error_msg))
        return false;
    if (!emit_vjp_region_ops(graph, bctx.forward_ctx(), region, error_msg))
        return false;

    for (std::size_t k = 0; k < region.input_ids.size() && k < region.input_grad_ids.size(); ++k) {
        const TensorId in_id = region.input_ids[k];
        const TensorId gin_id = region.input_grad_ids[k];
        if (in_id >= 0 && gin_id >= 0 && depends.count(in_id) != 0) {
            void* gin = bctx.forward(gin_id);
            if (gin == nullptr) {
                if (error_msg)
                    *error_msg = "manual_vjp: custom backward input grad id " +
                                 std::to_string(gin_id) + " has no binding";
                return false;
            }
            bctx.accumulate_grad(in_id, gin);
        }
    }
    return true;
}

}  // namespace

void register_vjp_emitter(std::unique_ptr<VjpEmitter> vjp) {
    if (!vjp)
        return;
    auto& r = vjp_registry();
    std::lock_guard<std::mutex> lk(r.mu);
    std::string key{vjp->op_name()};
    r.map[std::move(key)] = std::move(vjp);
}

VjpEmitter* find_vjp_emitter(std::string_view op_name) {
    auto& r = vjp_registry();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.map.find(std::string(op_name));
    return it == r.map.end() ? nullptr : it->second.get();
}

// ────────────────────────────────────────────────────────────────────
// Env-var gates
// ────────────────────────────────────────────────────────────────────

namespace {
bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr)
        return false;
    if (*v == '\0')
        return false;
    return !(v[0] == '0' && v[1] == '\0');
}
}  // namespace

bool use_manual_vjp() {
    // Production default = ON.  Manual VJP runs first for every op
    // we have an emitter for; coverage gaps soft-fall through to
    // ``gradientForPrimaryTensor:`` (the legacy path) so behaviour
    // stays at-least-as-good as the pre-manual-VJP baseline.
    //
    // Explicit opt-out: ``LUCID_MANUAL_VJP=0``.  Set this when you
    // need bit-exact reproduction of a prior eager-autograd baseline
    // or are debugging a manual-VJP regression.
    const char* v = std::getenv("LUCID_MANUAL_VJP");
    if (v == nullptr)
        return true;  // default ON
    return env_truthy("LUCID_MANUAL_VJP");
}

bool use_manual_vjp_require() {
    return env_truthy("LUCID_MANUAL_VJP_REQUIRE");
}

bool use_manual_vjp_debug() {
    return env_truthy("LUCID_MANUAL_VJP_DEBUG");
}

VjpRegistration vjp_registration_status(const std::string& op_name) {
    {
        auto& r = vjp_registry();
        std::lock_guard<std::mutex> lk(r.mu);
        if (r.map.find(op_name) != r.map.end())
            return VjpRegistration::Registered;
    }
    const auto& sinks = no_grad_ops();
    if (sinks.find(op_name) != sinks.end())
        return VjpRegistration::GradSink;
    return VjpRegistration::Missing;
}

// ────────────────────────────────────────────────────────────────────
// Error formatting helpers — shared by the walker's failure paths so
// debug output stays consistent with the propagated error string.
// ────────────────────────────────────────────────────────────────────

namespace {

std::string format_shape(const std::vector<std::int64_t>& shape) {
    if (shape.empty())
        return "()";
    std::ostringstream os;
    os << '(';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size())
            os << ", ";
    }
    os << ')';
    return os.str();
}

std::string format_node_signature(const TraceGraph& trace, const OpNode& node) {
    std::ostringstream os;
    os << "op='" << node.name << "'";
    os << " inputs=[";
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
        const TensorId tid = node.inputs[i];
        os << format_shape(shape_of_tid(trace, tid)) << ':'
           << std::string(dtype_name(dtype_of_tid(trace, tid)));
        if (i + 1 < node.inputs.size())
            os << ", ";
    }
    os << "] outputs=[";
    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
        os << format_shape(node.outputs[i].shape) << ':'
           << std::string(dtype_name(node.outputs[i].dtype));
        if (i + 1 < node.outputs.size())
            os << ", ";
    }
    os << "]";
    return os.str();
}

void debug_log_failure(const std::string& reason, const std::string& signature) {
    if (!use_manual_vjp_debug())
        return;
    std::cerr << "[lucid.compile manual_vjp] " << reason << '\n'
              << "[lucid.compile manual_vjp]   " << signature << '\n'
              << "[lucid.compile manual_vjp]   verdict: "
              << (use_manual_vjp_require() ? "HARD-FAIL (LUCID_MANUAL_VJP_REQUIRE=1)"
                                           : "soft-fallback to gradientForPrimaryTensor:")
              << std::endl;
}

}  // namespace

ManualVjpStatus try_manual_vjp_grads(void* graph_void,
                                     BuilderContext& fwd,
                                     const TraceGraph& trace,
                                     TensorId loss_id,
                                     const std::vector<TensorId>& param_ids,
                                     std::vector<void*>& out_grads,
                                     std::string* error_msg,
                                     std::vector<bool>* absent) {
    if (!use_manual_vjp())
        return ManualVjpStatus::Disabled;
    BackwardWalker walker{graph_void, fwd, trace};
    std::string vjp_err;
    if (walker.compute_grads(loss_id, param_ids, out_grads, &vjp_err, absent))
        return ManualVjpStatus::Success;
    if (error_msg)
        *error_msg = std::move(vjp_err);
    return use_manual_vjp_require() ? ManualVjpStatus::HardFail : ManualVjpStatus::FellBack;
}

// ────────────────────────────────────────────────────────────────────
// BackwardContext
// ────────────────────────────────────────────────────────────────────

BackwardContext::BackwardContext(void* graph_void, BuilderContext& fwd)
    : graph_(graph_void), fwd_(fwd) {}

void* BackwardContext::forward(TensorId tid) const {
    return fwd_.resolve(tid);
}

void* BackwardContext::resolve_grad(TensorId tid) const {
    auto it = grad_map_.find(tid);
    return it == grad_map_.end() ? nullptr : it->second;
}

void BackwardContext::accumulate_grad(TensorId tid, void* grad) {
    if (grad == nullptr)
        return;
    MPSGraph* g = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* b = (__bridge MPSGraphTensor*)grad;
    MPSGraphTensor* fwd = (__bridge MPSGraphTensor*)forward(tid);
    // Match eager autograd's ``grad_to_input_dtype``: each tensor id's
    // gradient lives in that tensor's own forward dtype.
    if (g != nil && b != nil && fwd != nil && (fwd.dataType & MPSDataTypeFloatBit) != 0 &&
        b.dataType != fwd.dataType) {
        b = cast_mps_tensor(g, b, fwd.dataType, @"accum_to_fwd_dtype");
        grad = (__bridge void*)b;
    }
    auto it = grad_map_.find(tid);
    if (it == grad_map_.end()) {
        grad_map_[tid] = grad;
        return;
    }
    // Sum existing contribution + new one.
    MPSGraphTensor* a = (__bridge MPSGraphTensor*)it->second;
    if (a != nil && b != nil && a.dataType != b.dataType) {
        b = cast_mps_tensor(g, b, a.dataType, @"grad_accum_cast");
    }
    MPSGraphTensor* s = [g additionWithPrimaryTensor:a secondaryTensor:b name:@"grad_accum"];
    it->second = (__bridge void*)s;
}

void* BackwardContext::unreduce(void* grad,
                                const std::vector<std::int64_t>& target_shape,
                                const std::vector<std::int64_t>& broadcast_shape) {
    if (grad == nullptr)
        return nullptr;
    MPSGraph* g = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* t = (__bridge MPSGraphTensor*)grad;
    return (__bridge void*)unreduce_impl(g, t, broadcast_shape, target_shape);
}

// ────────────────────────────────────────────────────────────────────
// BackwardWalker
// ────────────────────────────────────────────────────────────────────

namespace {

bool emit_backward_node(MPSGraph* graph,
                        BackwardContext& bctx,
                        const TraceGraph& trace,
                        const OpNode& node,
                        const std::unordered_set<TensorId>& depends,
                        std::string* error_msg) {
    bool upstream_param = false;
    for (const auto& om : node.outputs) {
        if (depends.count(om.id) != 0) {
            upstream_param = true;
            break;
        }
    }
    if (!upstream_param)
        return true;

    std::vector<void*> grad_outs;
    grad_outs.reserve(node.outputs.size());
    bool any_demand = false;
    for (const auto& om : node.outputs) {
        void* go = bctx.resolve_grad(om.id);
        if (go != nullptr) {
            any_demand = true;
            if (auto dt = mps_dtype_of(om.dtype))
                go = from_tensor(cast_if_needed(graph, as_tensor(go), *dt));
        }
        grad_outs.push_back(go);
    }
    const auto& sink = no_grad_ops();
    if (!any_demand || sink.find(node.name) != sink.end() || !carries_gradient(node))
        return true;

    VjpEmitter* vjp = find_vjp_emitter(node.name);
    if (vjp == nullptr) {
        std::string sig = format_node_signature(trace, node);
        debug_log_failure("no VJP emitter for op '" + node.name + "'", sig);
        if (error_msg)
            *error_msg = "manual_vjp: no VJP emitter for op '" + node.name + "' (" + sig + ")";
        return false;
    }
    if (!vjp->emit(bctx, node, grad_outs)) {
        std::string sig = format_node_signature(trace, node);
        debug_log_failure("VJP for op '" + node.name + "' returned false", sig);
        if (error_msg)
            *error_msg = "manual_vjp: VJP for op '" + node.name + "' returned false (" + sig + ")";
        return false;
    }
    return true;
}

}  // namespace

BackwardWalker::BackwardWalker(void* graph_void, BuilderContext& fwd, const TraceGraph& trace)
    : graph_(graph_void), trace_(trace), bctx_(graph_void, fwd) {}

bool BackwardWalker::compute_grads(TensorId loss_id,
                                   const std::vector<TensorId>& param_ids,
                                   std::vector<void*>& out_grads,
                                   std::string* error_msg,
                                   std::vector<bool>* absent) {
    auto set_err = [&](std::string msg) {
        if (error_msg)
            *error_msg = std::move(msg);
    };

    void* loss_void = bctx_.forward(loss_id);
    if (loss_void == nullptr) {
        set_err("manual_vjp: loss id " + std::to_string(loss_id) + " has no forward binding");
        return false;
    }
    auto loss_shape = shape_of_tid(trace_, loss_id);
    auto loss_dt = dtype_of_tid(trace_, loss_id);
    MPSGraph* graph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
    if (loss_shape.empty())
        loss_shape = shape_of_mps(loss_t);
    MPSGraphTensor* loss_grad = ones_like_loss(graph, loss_shape, loss_dt);
    bctx_.accumulate_grad(loss_id, (__bridge void*)loss_grad);

    const auto& ops = trace_.ops;
    std::unordered_map<std::size_t, const VjpRegion*> region_by_end;
    for (const auto& r : trace_.vjp_regions) {
        if (r.fwd_end > r.fwd_begin)
            region_by_end[r.fwd_end] = &r;
    }
    const std::unordered_set<TensorId> depends = param_dependents(trace_, param_ids);
    std::size_t idx = ops.size();
    while (idx > 0) {
        auto rit = region_by_end.find(idx);
        if (rit != region_by_end.end()) {
            const VjpRegion& region = *rit->second;
            if (!emit_vjp_region(graph, bctx_, region, depends, error_msg))
                return false;
            idx = region.fwd_begin;
            continue;
        }
        if (!emit_backward_node(graph, bctx_, trace_, ops[--idx], depends, error_msg))
            return false;
    }

    out_grads.clear();
    out_grads.reserve(param_ids.size());
    if (absent)
        absent->assign(param_ids.size(), false);
    for (std::size_t i = 0; i < param_ids.size(); ++i) {
        const TensorId pid = param_ids[i];
        void* g = bctx_.resolve_grad(pid);
        if (g == nullptr && absent) {
            MPSGraphTensor* p = (__bridge MPSGraphTensor*)bctx_.forward(pid);
            if (p != nil) {
                g = (__bridge void*)[graph constantWithScalar:0.0
                                                        shape:p.shape
                                                     dataType:p.dataType];
                (*absent)[i] = true;
            }
        }
        if (g == nullptr) {
            set_err("manual_vjp: no gradient produced for param id " + std::to_string(pid) +
                    " — most likely the param isn't on the loss path");
            return false;
        }
        out_grads.push_back(g);
    }
    return true;
}

}  // namespace lucid::compile
