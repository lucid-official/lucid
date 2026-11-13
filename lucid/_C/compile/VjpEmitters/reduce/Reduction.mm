// lucid/_C/compile/VjpEmitters/reduce/Reduction.mm
//
// VJPs for the ``sum`` / ``mean`` / ``max`` / ``min`` / ``p_norm`` reductions and ``cumsum``.
//
// Forward emitter (see :file:`OpEmitters/reduce/Reduction.mm`) carries
// ``dims`` (vector<int64>) + ``keepdim`` (bool) attrs and squeezes the
// output if keepdim=false.  Backward:
//
//   sum:  dA = broadcast(grad, in_shape) — with axis re-insertion when
//         keepdim=false.
//   mean: same as sum, then divided by the number of reduced elements
//         (∏ in_shape[d] over d in dims).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "../../OpEmitters/_ReduceCanon.h"
#include "../../OpEmitters/_ScaledPNorm.h"
#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// Re-insert squeezed reduction axes back into the grad tensor.  If
// the forward op had keepdim=false, grad arrives with rank
// ``in_rank - n_reduced``; we need to reshape it so each reduced
// axis is size 1 before broadcasting back to ``in_shape``.
MPSGraphTensor* unsqueeze_reduced(MPSGraph* g, MPSGraphTensor* grad,
                                  const std::vector<std::int64_t>& in_shape,
                                  const std::vector<std::int64_t>& dims) {
    // Build the keepdim-style shape: in_shape but with reduced axes
    // replaced by 1.
    std::vector<bool> is_reduced(in_shape.size(), false);
    for (std::int64_t d : dims) {
        std::int64_t ax = (d < 0) ? d + (std::int64_t)in_shape.size() : d;
        if (ax >= 0 && ax < (std::int64_t)in_shape.size())
            is_reduced[ax] = true;
    }
    NSMutableArray<NSNumber*>* shape =
        [NSMutableArray arrayWithCapacity:in_shape.size()];
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
        std::int64_t s = is_reduced[i] ? 1 : in_shape[i];
        [shape addObject:[NSNumber numberWithLongLong:s]];
    }
    return [g reshapeTensor:grad withShape:shape name:nil];
}

// ``grad`` (the reduction's output, ``keepdim`` or squeezed) spread back over
// the reduced axes of ``in_shape``, scaled by ``scale`` first when it is not 1.
// A rank-5+ input is spread at its canonical lower rank and reshaped back, so
// the broadcast never runs on a rank-5+ tensor.
MPSGraphTensor* spread_over_reduced(MPSGraph* g,
                                    MPSGraphTensor* grad,
                                    const std::vector<std::int64_t>& in_shape,
                                    const std::vector<std::int64_t>& dims,
                                    bool keepdim,
                                    double scale,
                                    NSString* name) {
    const ReduceWork work(in_shape,
                          reduce_canon_mask(in_shape.size(), dims)
                              .value_or(std::vector<bool>(in_shape.size(), false)));
    MPSGraphTensor* grad_keep = work.canonical()
                                    ? work.keep(g, grad)
                                    : (keepdim ? grad : unsqueeze_reduced(g, grad, in_shape, dims));
    if (scale != 1.0) {
        grad_keep = [g multiplicationWithPrimaryTensor:grad_keep
                                       secondaryTensor:[g constantWithScalar:scale
                                                                    dataType:grad_keep.dataType]
                                                  name:nil];
    }
    MPSGraphTensor* spread = [g broadcastTensor:grad_keep
                                        toShape:shape_to_ns(work.work_shape())
                                           name:work.canonical() ? nil : name];
    return work.canonical() ? [g reshapeTensor:spread withShape:shape_to_ns(in_shape) name:name]
                            : spread;
}

// Common ``dims`` + ``keepdim`` attr extraction for reduction VJPs.
// Returns ``{dims_ptr, keepdim_ptr}`` with one or both null on failure.
struct ReduceAttrs {
    const std::vector<std::int64_t>* dims = nullptr;
    const bool* keepdim = nullptr;
};

inline ReduceAttrs extract_reduce_attrs(const OpNode& node) {
    ReduceAttrs r;
    auto dims_it = node.attrs.find("dims");
    auto kd_it = node.attrs.find("keepdim");
    if (dims_it == node.attrs.end() || kd_it == node.attrs.end()) return r;
    r.dims = std::get_if<std::vector<std::int64_t>>(&dims_it->second);
    r.keepdim = std::get_if<bool>(&kd_it->second);
    return r;
}

class SumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sum"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        ReduceAttrs a = extract_reduce_attrs(node);
        if (a.dims == nullptr || a.keepdim == nullptr) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&a](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) -> MPSGraphTensor* {
                std::vector<std::int64_t> in_shape = shape_of_mps(x);
                // The sum of a 0-d tensor is that tensor; this used to
                // decline, and any trace summing a scalar — a loss built
                // from ``det`` or any other scalar — fell back.
                if (in_shape.empty()) return [g reshapeTensor:grad withShape:@[] name:@"sum_vjp"];
                return spread_over_reduced(g, grad, in_shape, *a.dims, *a.keepdim, 1.0, @"sum_vjp");
            });
    }
};

class MeanVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mean"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        ReduceAttrs a = extract_reduce_attrs(node);
        if (a.dims == nullptr || a.keepdim == nullptr) return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&a](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* grad) -> MPSGraphTensor* {
                std::vector<std::int64_t> in_shape = shape_of_mps(x);
                if (in_shape.empty()) return [g reshapeTensor:grad withShape:@[] name:@"mean_vjp"];
                // N = product of reduced-axis sizes.
                double N = 1.0;
                for (std::int64_t d : *a.dims) {
                    std::int64_t ax = (d < 0) ? d + (std::int64_t)in_shape.size() : d;
                    if (ax >= 0 && ax < (std::int64_t)in_shape.size())
                        N *= (double)in_shape[ax];
                }
                return spread_over_reduced(
                    g, grad, in_shape, *a.dims, *a.keepdim, 1.0 / N, @"mean_vjp");
            });
    }
};

// The axes the forward reduced, as the trace recorded them (none recorded =
// all); nullopt for an axis outside the input's rank.
std::optional<std::vector<bool>> reduced_axes(const OpNode& node, std::size_t rank) {
    auto it = node.attrs.find("dims");
    const auto* dims =
        it == node.attrs.end() ? nullptr : std::get_if<std::vector<std::int64_t>>(&it->second);
    if (dims == nullptr || dims->empty())
        return std::vector<bool>(rank, true);
    return reduce_canon_mask(rank, *dims);
}

// How a NaN slice's gradient is shaped: eager's ``ExtremumNanGrad``
// (ops/ufunc/Reductions.h), carried by the forward as the ``nan_grad`` attr —
// absent for the default.
enum class NanGrad : std::int64_t { Slice = 0, Element = 1, Zero = 2 };

NanGrad nan_grad_of(const OpNode& node) {
    auto it = node.attrs.find("nan_grad");
    const auto* v = it == node.attrs.end() ? nullptr : std::get_if<std::int64_t>(&it->second);
    if (v == nullptr || *v < 0 || *v > 2)
        return NanGrad::Slice;
    return static_cast<NanGrad>(*v);
}

// Float16 / bfloat16 count exactly only to 2048 / 256 tied elements, so the share
// is built in float32 (LCD-879).
MPSDataType tie_count_dtype(MPSDataType dt) {
    return (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16) ? MPSDataTypeFloat32 : dt;
}

// ``mask / denom`` with ``ties`` the sum of ``mask`` over ``axes``: eager's
// ``extremum_share_counted``.  ``Slice`` divides by ``ties`` itself, so a slice nothing
// matches (a NaN extremum) is 0 / 0 = NaN at every element; ``Element`` first
// widens ``mask`` to the NaN elements of a NaN-extremum slice; ``Zero`` divides
// by ``ties + 1 - mask``, so that slice gets zeros.  ``x`` and ``y`` (the
// result, in keepdim form) are only read to find the NaN elements.  An integer
// input holds no NaN and takes ``Zero``'s form, which is the same there.
MPSGraphTensor* share_among_ties(MPSGraph* g,
                                 MPSGraphTensor* mask,
                                 NSArray<NSNumber*>* axes,
                                 NanGrad rule,
                                 MPSGraphTensor* x,
                                 MPSGraphTensor* y) {
    const bool nan_aware = rule != NanGrad::Zero && (x.dataType & MPSDataTypeFloatBit) != 0;
    if (nan_aware && rule == NanGrad::Element) {
        MPSGraphTensor* both_nan = [g logicalANDWithPrimaryTensor:[g isNaNWithTensor:x name:nil]
                                                  secondaryTensor:[g isNaNWithTensor:y name:nil]
                                                             name:nil];
        mask = [g additionWithPrimaryTensor:mask
                            secondaryTensor:[g castTensor:both_nan toType:mask.dataType name:nil]
                                       name:nil];
    }
    // A 0-d input reduces over no axis: its one element is its own tie count.
    MPSGraphTensor* ties =
        axes.count == 0 ? mask : [g reductionSumWithTensor:mask axes:axes name:nil];
    if (nan_aware && rule == NanGrad::Slice)
        return [g divisionWithPrimaryTensor:mask secondaryTensor:ties name:nil];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:mask.dataType];
    MPSGraphTensor* denom = [g subtractionWithPrimaryTensor:[g additionWithPrimaryTensor:ties
                                                                         secondaryTensor:one
                                                                                    name:nil]
                                            secondaryTensor:mask
                                                       name:nil];
    return [g divisionWithPrimaryTensor:mask secondaryTensor:denom name:nil];
}

// What a one-input reduction's VJP reads: the graph, the gradient of the result (``go``),
// the input (``x``) and the result itself (``y``).  False from ``open_reduce_vjp`` when the
// node or the forward leaves one of them out.
struct ReduceVjpOperands {
    MPSGraph* g = nil;
    MPSGraphTensor* go = nil;
    MPSGraphTensor* x = nil;
    MPSGraphTensor* y = nil;
};

bool open_reduce_vjp(BackwardContext& bctx,
                     const OpNode& node,
                     const std::vector<void*>& grad_outs,
                     ReduceVjpOperands& ops) {
    if (node.inputs.size() != 1 || node.inputs[0] < 0 || node.outputs.empty() ||
        grad_outs.empty() || grad_outs[0] == nullptr)
        return false;
    ops.g = (__bridge MPSGraph*)bctx.graph();
    ops.go = as_tensor(grad_outs[0]);
    ops.x = as_tensor(bctx.forward(node.inputs[0]));
    ops.y = as_tensor(bctx.forward(node.outputs[0].id));
    return ops.g != nil && ops.go != nil && ops.x != nil && ops.y != nil;
}

// ────────────────────────────────────────────────────────────────────
// max / min: a slice's gradient is shared evenly among the elements tied
// for its extremum, ``mask / (ties + 1 - mask)`` with ``mask = (x == y)``
// — eager's ``extremum_grad`` (ops/ufunc/Reductions.cpp).  A half dtype's tie
// count is exact only to 256 (bfloat16) / 2048 (float16), so the mask, the count
// and the share are taken in float32 and the product rounds once, as eager's
// does (LCD-879).
// A slice holding a NaN is shaped by the forward's ``nan_grad`` attr (see
// ``share_among_ties``): NaN everywhere (``amax``), the NaN elements
// (``max()``) or zeros (``vector_norm(inf)``).  Giving every tie the whole gradient made a step on a tie — a
// zero-initialised tensor, ``matrix_norm(a, 2)`` or ``cond`` on a repeated
// singular value — that many times too large (LCD-865).  An int ``dim``
// (``x.max(dim)``) gathers at its argmax and never reaches this VJP.
// cumsum: the adjoint of a running sum is the running sum taken backwards.
// Neither had a VJP, and MPSGraph's autodiff aborts on both, so
// ``logsumexp``, ``norm`` and any max over a sequence ran eager.
// ────────────────────────────────────────────────────────────────────
template <bool IS_MAX>
class ExtremumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return IS_MAX ? "max" : "min"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        ReduceVjpOperands ops;
        if (!open_reduce_vjp(bctx, node, grad_outs, ops))
            return false;
        MPSGraph* g = ops.g;
        MPSGraphTensor* go = ops.go;
        MPSGraphTensor* x = ops.x;
        MPSGraphTensor* y = ops.y;
        const auto x_shape = shape_of_mps(x);
        const auto reduced = reduced_axes(node, x_shape.size());
        if (!reduced)
            return false;
        // Result and grad in keepdim form, so both broadcast against x.  A rank-5+
        // input is worked at its canonical lower rank (LCD-918): the mask, the
        // tie count and the share are all elementwise over rank-5+ tensors
        // otherwise.
        const ReduceWork work(x_shape, *reduced);
        MPSGraphTensor* x_e = work.enter(g, x);
        MPSGraphTensor* y_k = work.keep(g, cast_if_needed(g, y, x.dataType));
        MPSGraphTensor* go_k = work.keep(g, go);
        const MPSDataType count_dt = tie_count_dtype(go.dataType);
        MPSGraphTensor* mask = [g castTensor:[g equalWithPrimaryTensor:x_e
                                                       secondaryTensor:y_k
                                                                  name:nil]
                                      toType:count_dt
                                        name:nil];
        MPSGraphTensor* share =
            share_among_ties(g, mask, work.axes_ns(), nan_grad_of(node), x_e, y_k);
        MPSGraphTensor* scaled =
            [g multiplicationWithPrimaryTensor:cast_if_needed(g, go_k, count_dt)
                               secondaryTensor:share
                                          name:IS_MAX ? @"max_vjp" : @"min_vjp"];
        MPSGraphTensor* dx = work.leave(g, cast_if_needed(g, scaled, go.dataType));
        bctx.accumulate_grad(node.inputs[0], from_tensor(dx));
        return true;
    }
};

class CumsumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "cumsum"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 1 || node.inputs[0] < 0 || grad_outs.empty() ||
            grad_outs[0] == nullptr)
            return false;
        auto it = node.attrs.find("axis");
        if (it == node.attrs.end())
            return false;
        const auto* ax = std::get_if<std::int64_t>(&it->second);
        if (ax == nullptr)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* dx = [g cumulativeSumWithTensor:go
                                                   axis:(NSInteger)*ax
                                              exclusive:NO
                                                reverse:YES
                                                   name:@"cumsum_vjp"];
        bctx.accumulate_grad(node.inputs[0], from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// p_norm (finite p > 1) — the engine's ``PNormBackward``:
//
//   dA = sign(A) * (|A| / N)^(p-1) * expand(dN),   0 where N == 0
//
// ``|A| / N`` is in [0, 1], so the slope is as range-safe as the forward
// (``_ScaledPNorm.h``): a slice at 1e-25 gets the gradient a slice at 1 does.
// A zero norm has the subgradient 0 (its ratio is 0 / 0); a NaN norm is not
// zero and keeps its NaN.  Without this VJP the op would be differentiated by
// MPSGraph's autodiff, which aborts on the forward's selects.
// ────────────────────────────────────────────────────────────────────
class PNormVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "p_norm"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        ReduceVjpOperands ops;
        if (!open_reduce_vjp(bctx, node, grad_outs, ops))
            return false;
        MPSGraph* g = ops.g;
        const auto x_shape = shape_of_mps(ops.x);
        const auto spec = p_norm_spec(node, x_shape);
        if (!spec)
            return false;
        // The chain's dtype, as every VJP (autocast reconciliation); a rank-5+
        // input is worked at its canonical lower rank (LCD-918).
        const MPSDataType dt = ops.go.dataType;
        const ReduceWork work(x_shape, spec->reduced);
        MPSGraphTensor* dx = work.leave(
            g, p_norm_grad(g, work.enter(g, cast_if_needed(g, ops.x, dt)),
                           work.keep(g, cast_if_needed(g, ops.y, dt)), work.keep(g, ops.go),
                           spec->p));
        bctx.accumulate_grad(node.inputs[0], from_tensor(dx));
        return true;
    }
};

struct ReductionVjpRegistrar {
    ReductionVjpRegistrar() {
        register_vjp_emitter(std::make_unique<SumVjp>());
        register_vjp_emitter(std::make_unique<ExtremumVjp<true>>());
        register_vjp_emitter(std::make_unique<ExtremumVjp<false>>());
        register_vjp_emitter(std::make_unique<CumsumVjp>());
        register_vjp_emitter(std::make_unique<MeanVjp>());
        register_vjp_emitter(std::make_unique<PNormVjp>());
    }
};

[[maybe_unused]] static const ReductionVjpRegistrar g_reduction_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
