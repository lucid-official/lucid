// lucid/_C/compile/VjpEmitters/reduce/Reduction.mm
//
// VJPs for the ``sum`` / ``mean`` / ``max`` / ``min`` reductions and ``cumsum``.
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
                MPSGraphTensor* grad_keep =
                    *a.keepdim ? grad : unsqueeze_reduced(g, grad, in_shape, *a.dims);
                return [g broadcastTensor:grad_keep
                                  toShape:shape_to_ns(in_shape)
                                     name:@"sum_vjp"];
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
                MPSGraphTensor* grad_keep =
                    *a.keepdim ? grad : unsqueeze_reduced(g, grad, in_shape, *a.dims);
                MPSGraphTensor* inv_n = [g constantWithScalar:(1.0 / N)
                                                     dataType:grad_keep.dataType];
                MPSGraphTensor* scaled =
                    [g multiplicationWithPrimaryTensor:grad_keep
                                       secondaryTensor:inv_n name:nil];
                return [g broadcastTensor:scaled
                                  toShape:shape_to_ns(in_shape)
                                     name:@"mean_vjp"];
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
    std::vector<bool> reduced(rank, false);
    const auto r = static_cast<std::int64_t>(rank);
    for (std::int64_t d : *dims) {
        if (d < 0)
            d += r;
        if (d < 0 || d >= r)
            return std::nullopt;
        reduced[static_cast<std::size_t>(d)] = true;
    }
    return reduced;
}

// ``mask / (ties + 1 - mask)``, with ``ties`` the sum of ``mask`` over ``axes``.
MPSGraphTensor* share_among_ties(MPSGraph* g, MPSGraphTensor* mask, NSArray<NSNumber*>* axes) {
    // A 0-d input reduces over no axis: its one element is its own tie count.
    MPSGraphTensor* ties =
        axes.count == 0 ? mask : [g reductionSumWithTensor:mask axes:axes name:nil];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:mask.dataType];
    MPSGraphTensor* denom = [g subtractionWithPrimaryTensor:[g additionWithPrimaryTensor:ties
                                                                         secondaryTensor:one
                                                                                    name:nil]
                                            secondaryTensor:mask
                                                       name:nil];
    return [g divisionWithPrimaryTensor:mask secondaryTensor:denom name:nil];
}

// ────────────────────────────────────────────────────────────────────
// max / min: a slice's gradient is shared evenly among the elements tied
// for its extremum, ``mask / (ties + 1 - mask)`` with ``mask = (x == y)``
// — eager's ``share_among_ties`` (ops/ufunc/Reductions.cpp), in the same
// order of operations so a half-precision share rounds as eager's does while
// the tie count is exact in that dtype (bfloat16 up to 256, float16 up to
// 2048); past that eager's ``ties + 1`` rounds and the two differ.
// Where ``mask`` is 1 the denominator is the slice's tie count; where it
// is 0 it is at least 1, so a slice nothing matches (a NaN extremum) gets
// zeros.  Giving every tie the whole gradient made a step on a tie — a
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
        if (node.inputs.size() != 1 || node.inputs[0] < 0 || node.outputs.empty() ||
            grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(node.inputs[0]));
        MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
        if (g == nil || go == nil || x == nil || y == nil)
            return false;
        const auto x_shape = shape_of_mps(x);
        const auto reduced = reduced_axes(node, x_shape.size());
        if (!reduced)
            return false;
        // Result and grad in keepdim form, so both broadcast against x.
        std::vector<std::int64_t> kept(x_shape);
        NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:x_shape.size()];
        for (std::size_t d = 0; d < kept.size(); ++d) {
            if (!(*reduced)[d])
                continue;
            kept[d] = 1;
            [axes addObject:[NSNumber numberWithUnsignedLong:d]];
        }
        MPSGraphTensor* y_k = [g reshapeTensor:cast_if_needed(g, y, x.dataType)
                                     withShape:shape_to_ns(kept)
                                          name:nil];
        MPSGraphTensor* go_k = [g reshapeTensor:go withShape:shape_to_ns(kept) name:nil];
        MPSGraphTensor* mask = [g castTensor:[g equalWithPrimaryTensor:x
                                                       secondaryTensor:y_k
                                                                  name:nil]
                                      toType:go.dataType
                                        name:nil];
        MPSGraphTensor* dx = [g multiplicationWithPrimaryTensor:go_k
                                                secondaryTensor:share_among_ties(g, mask, axes)
                                                           name:IS_MAX ? @"max_vjp" : @"min_vjp"];
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

struct ReductionVjpRegistrar {
    ReductionVjpRegistrar() {
        register_vjp_emitter(std::make_unique<SumVjp>());
        register_vjp_emitter(std::make_unique<ExtremumVjp<true>>());
        register_vjp_emitter(std::make_unique<ExtremumVjp<false>>());
        register_vjp_emitter(std::make_unique<CumsumVjp>());
        register_vjp_emitter(std::make_unique<MeanVjp>());
    }
};

[[maybe_unused]] static const ReductionVjpRegistrar g_reduction_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
