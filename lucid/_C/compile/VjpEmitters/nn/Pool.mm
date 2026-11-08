// lucid/_C/compile/VjpEmitters/nn/Pool.mm
//
// Pool VJPs — {max, avg}_pool{1d, 2d, 3d}.
//
// Critical for ResNet-class workloads: the manual VJP walker
// previously fell back to MPSGraph autograd for these ops, which
// breaks under autocast (gradientForPrimaryTensor: emits constants
// in F32 regardless of the F16 input chain → MPSGraph type-check
// failure deep in MLIR pass manager).  Manual VJP keeps the chain
// in a single dtype.
//
// Every descriptor is built the way its forward emitter builds it — ceil
// mode, and for averages the settled divisor (``settle_ceil_divisor``).
// The 2-D average VJP used to hard-wire ``includeZeroPadToAverage`` and
// ignore ceil mode, so once the forward honoured both, a compiled step
// through ``count_include_pad=False`` took its gradient with the wrong
// divisor.  1-D pools are 2-D pools of height 1 and 3-D pools 4-D pools of
// a leading length-1 axis, exactly as the forward lifts them; neither had
// a VJP, and a U-Net in 3-D or a 1-D audio net trained eager.
//
//   avg: MPSGraph's pooling-gradient kernel, same descriptor.
//   max: the arg-max found by comparing strided slices of the padded source
//        with the pooled value (``local_argmax``), then a scatter-add of the
//        gradient to those positions.  ``maxPooling2DGradientWithGradientTensor``
//        placed the gradient wrongly when it arrived transposed — a pooled
//        map reshaped to a sequence and permuted, as CoAtNet's downsampling
//        attention does, came back 50–100 % off against both eager and an
//        independent reference — and the indices-based gradient variant
//        crashed on the same input.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "../_VjpHelpers.h"
#include "../../OpEmitters/_AttrHelpers.h"
#include "../../OpEmitters/_MaxPool.h"

namespace lucid::compile {

namespace {

// Pull a list-of-int attr or return nullptr.
inline const std::vector<std::int64_t>* int_vec_attr_local(
    const OpNode& node, const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end()) return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

// The trailing ``nd`` extents of ``shape``.
std::vector<std::int64_t> trailing_extent(NSArray<NSNumber*>* shape, std::size_t nd) {
    std::vector<std::int64_t> out;
    for (NSUInteger i = shape.count - nd; i < shape.count; ++i)
        out.push_back(shape[i].longLongValue);
    return out;
}

bool known(NSArray<NSNumber*>* shape) {
    for (NSNumber* n in shape)
        if (n.longLongValue < 0)
            return false;
    return true;
}

// Scatter ``grad`` to the arg-max positions of a zero tensor shaped like
// ``x``.  ``local`` holds each window's arg-max as a flat index *within the
// window* (``LocalFlatten``); the position in ``x`` is the window's origin
// plus that offset.  Global flat indices went wrong once MPSGraph folded a
// preceding pad into the pooling: they then counted in the unpadded tensor
// while the scatter counted in the padded one — YOLOv3-tiny's pad-then-pool
// came back 87 % off.  A window-local index is the same either way.
MPSGraphTensor* scatter_to_argmax(MPSGraph* g, MPSGraphTensor* grad, MPSGraphTensor* local,
                                  NSArray<NSNumber*>* xs, NSArray<NSNumber*>* gs,
                                  const std::vector<std::int64_t>& kernel,
                                  const std::vector<std::int64_t>& stride,
                                  const std::vector<std::int64_t>& pad_front) {
    const NSUInteger nd = xs.count - 2;
    MPSGraphTensor* rem = [g castTensor:local toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* flat = nil;
    // Decompose the window-local index, last axis fastest, and accumulate
    // the global flat position over the spatial axes of ``x``.
    std::vector<MPSGraphTensor*> pos(nd, nil);
    for (NSInteger a = (NSInteger)nd - 1; a >= 0; --a) {
        MPSGraphTensor* k = [g constantWithScalar:(double)kernel[(std::size_t)a]
                                         dataType:MPSDataTypeInt32];
        MPSGraphTensor* off = [g moduloWithPrimaryTensor:rem secondaryTensor:k name:nil];
        // Exact: ``rem - off`` is a multiple of ``k``.
        rem = [g divisionWithPrimaryTensor:[g subtractionWithPrimaryTensor:rem
                                                           secondaryTensor:off
                                                                      name:nil]
                           secondaryTensor:k
                                      name:nil];
        MPSGraphTensor* o = [g coordinateAlongAxis:(NSInteger)(a + 2) withShape:gs name:nil];
        o = [g castTensor:o toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* origin = [g subtractionWithPrimaryTensor:
                                        [g multiplicationWithPrimaryTensor:o
                                                           secondaryTensor:[g constantWithScalar:(double)stride[(std::size_t)a]
                                                                                        dataType:MPSDataTypeInt32]
                                                                      name:nil]
                                                 secondaryTensor:[g constantWithScalar:(double)pad_front[(std::size_t)a]
                                                                              dataType:MPSDataTypeInt32]
                                                            name:nil];
        pos[(std::size_t)a] = [g additionWithPrimaryTensor:origin secondaryTensor:off name:nil];
    }
    for (NSUInteger a = 0; a < nd; ++a) {
        MPSGraphTensor* extent = [g constantWithScalar:(double)xs[a + 2].longLongValue
                                              dataType:MPSDataTypeInt32];
        flat = flat == nil ? pos[a]
                           : [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:flat
                                                                            secondaryTensor:extent
                                                                                       name:nil]
                                         secondaryTensor:pos[a]
                                                    name:nil];
    }
    long long in_sp = 1, out_sp = 1;
    for (NSUInteger i = 2; i < xs.count; ++i)
        in_sp *= xs[i].longLongValue;
    for (NSUInteger i = 2; i < gs.count; ++i)
        out_sp *= gs[i].longLongValue;
    MPSGraphTensor* i3 = [g reshapeTensor:flat withShape:@[ gs[0], gs[1], @(out_sp) ] name:nil];
    MPSGraphTensor* u3 = [g reshapeTensor:grad withShape:@[ gs[0], gs[1], @(out_sp) ] name:nil];
    MPSGraphTensor* base = [g constantWithScalar:0.0
                                           shape:@[ xs[0], xs[1], @(in_sp) ]
                                        dataType:grad.dataType];
    MPSGraphTensor* out = [g scatterAlongAxis:2
                               withDataTensor:base
                                updatesTensor:u3
                                indicesTensor:i3
                                         mode:MPSGraphScatterModeAdd
                                         name:nil];
    return [g reshapeTensor:out withShape:xs name:nil];
}

// Each window's arg-max as a flat index within the window (last axis
// fastest), found without MPSGraph's pooling-with-indices kernel: the
// source is padded with NaN, and for every kernel offset the strided slice
// it picks out is compared with the pooled value ``y``; the first offset
// that matches wins, as eager's first-maximum rule has it.  The kernel is
// avoided because it aborted on macOS 15 ("Multi destination pooling
// encode has no non-nil destinations") under a fused, float16 training
// step — and it had already miscounted once MPSGraph folded a pad into it.
MPSGraphTensor* local_argmax(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* y,
                             const std::vector<std::int64_t>& kernel,
                             const std::vector<std::int64_t>& stride,
                             const std::vector<std::int64_t>& pad_front) {
    NSArray<NSNumber*>* xs = x.shape;
    NSArray<NSNumber*>* ys = y.shape;
    const std::size_t nd = xs.count - 2;
    NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithArray:@[ @0, @0 ]];
    NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithArray:@[ @0, @0 ]];
    for (std::size_t a = 0; a < nd; ++a) {
        const long long in = xs[a + 2].longLongValue, out = ys[a + 2].longLongValue;
        const long long need = (out - 1) * stride[a] + kernel[a] - in - pad_front[a];
        [left addObject:@(pad_front[a])];
        [right addObject:@(need > 0 ? need : 0)];
    }
    // Padding is NaN, which equals nothing: with ``-inf`` it would match a window of ``-inf``
    // (``y == -inf``) and the gradient would land on a padded position (LCD-916).
    MPSGraphTensor* xp = pad_or_identity(g, x, left, right, MPSGraphPaddingModeConstant, NAN);
    long long offsets = 1;
    for (std::size_t a = 0; a < nd; ++a)
        offsets *= kernel[a];
    MPSGraphTensor* found = [g constantWithScalar:0.0 shape:ys dataType:MPSDataTypeFloat32];
    MPSGraphTensor* local = [g constantWithScalar:0.0 shape:ys dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    for (long long t = 0; t < offsets; ++t) {
        std::vector<long long> off(nd, 0);
        long long rem = t;
        for (std::size_t a = nd; a-- > 0;) {
            off[a] = rem % kernel[a];
            rem /= kernel[a];
        }
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithArray:@[ @0, @0 ]];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithArray:@[ xs[0], xs[1] ]];
        NSMutableArray<NSNumber*>* steps = [NSMutableArray arrayWithArray:@[ @1, @1 ]];
        for (std::size_t a = 0; a < nd; ++a) {
            const long long out = ys[a + 2].longLongValue;
            [starts addObject:@(off[a])];
            [ends addObject:@(off[a] + (out - 1) * stride[a] + 1)];
            [steps addObject:@(stride[a])];
        }
        MPSGraphTensor* xt = [g sliceTensor:xp starts:starts ends:ends strides:steps name:nil];
        MPSGraphTensor* eq = [g castTensor:[g equalWithPrimaryTensor:xt secondaryTensor:y name:nil]
                                    toType:MPSDataTypeFloat32
                                      name:nil];
        MPSGraphTensor* hit = [g multiplicationWithPrimaryTensor:eq
                                                 secondaryTensor:[g subtractionWithPrimaryTensor:one
                                                                                 secondaryTensor:found
                                                                                            name:nil]
                                                            name:nil];
        found = [g additionWithPrimaryTensor:found secondaryTensor:hit name:nil];
        local = [g additionWithPrimaryTensor:local
                             secondaryTensor:[g multiplicationWithPrimaryTensor:hit
                                                                secondaryTensor:[g constantWithScalar:(double)t
                                                                                             dataType:MPSDataTypeFloat32]
                                                                           name:nil]
                                        name:nil];
    }
    return [g castTensor:local toType:MPSDataTypeInt32 name:nil];
}

// The pooled value the gradient is routed by: the forward's own when the walk holds it,
// which saves one pooling, else ``recompute()``.
template <class Recompute>
MPSGraphTensor* pooled_value(const BackwardContext& bctx,
                             const OpNode& node,
                             MPSDataType dtype,
                             NSArray<NSNumber*>* shape,
                             Recompute recompute) {
    if (node.outputs.empty() || bctx.forward(node.outputs[0].id) == nullptr)
        return recompute();
    MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
    MPSGraphTensor* y = cast_if_needed(g, as_tensor(bctx.forward(node.outputs[0].id)), dtype);
    return [g reshapeTensor:y withShape:shape name:nil];
}

// The pooled value recomputed when the walk holds no forward one, shaped ``lifted`` as the
// gradient kernels see it: the exact max pool the forward emitter builds over the unlifted ``x``
// (an all -inf window stays -inf, LCD-916), else ``plain()``.
template <class Plain>
MPSGraphTensor* recomputed_max_pool(MPSGraph* g,
                                    MPSGraphTensor* x,
                                    const std::vector<std::int64_t>& kernel,
                                    const std::vector<std::int64_t>& stride,
                                    const std::vector<std::int64_t>& pad,
                                    NSArray<NSNumber*>* pooled,
                                    NSArray<NSNumber*>* lifted,
                                    Plain plain) {
    MPSGraphTensor* exact =
        exact_max_pool(g, x, kernel, stride, pad, trailing_extent(pooled, kernel.size()));
    return exact != nil ? [g reshapeTensor:exact withShape:lifted name:nil] : plain();
}

MPSGraphTensor* recomputed_max_pool2d(MPSGraph* g,
                                      MPSGraphTensor* x,
                                      MPSGraphTensor* lifted,
                                      MPSGraphPooling2DOpDescriptor* d,
                                      const std::vector<std::int64_t>& kernel,
                                      const std::vector<std::int64_t>& stride,
                                      const std::vector<std::int64_t>& pad,
                                      NSArray<NSNumber*>* pooled,
                                      NSArray<NSNumber*>* pooled_lifted) {
    return recomputed_max_pool(g, x, kernel, stride, pad, pooled, pooled_lifted, [&] {
        return [g maxPooling2DWithSourceTensor:lifted descriptor:d name:nil];
    });
}

MPSGraphTensor* recomputed_max_pool4d(MPSGraph* g,
                                      MPSGraphTensor* x,
                                      MPSGraphTensor* lifted,
                                      MPSGraphPooling4DOpDescriptor* d,
                                      const std::vector<std::int64_t>& kernel,
                                      const std::vector<std::int64_t>& stride,
                                      const std::vector<std::int64_t>& pad,
                                      NSArray<NSNumber*>* pooled,
                                      NSArray<NSNumber*>* pooled_lifted) {
    return recomputed_max_pool(g, x, kernel, stride, pad, pooled, pooled_lifted, [&] {
        return [g maxPooling4DWithSourceTensor:lifted descriptor:d name:nil];
    });
}

// The gradient of a max pool: ``grad`` scattered to each window's first maximum.  ``nil`` if
// the pooled value ``y`` is.
MPSGraphTensor* max_pool_grad(MPSGraph* g,
                              MPSGraphTensor* x,
                              MPSGraphTensor* y,
                              MPSGraphTensor* grad,
                              NSArray<NSNumber*>* xs,
                              NSArray<NSNumber*>* gs,
                              const std::vector<std::int64_t>& kernel,
                              const std::vector<std::int64_t>& stride,
                              const std::vector<std::int64_t>& pad) {
    if (y == nil)
        return nil;
    MPSGraphTensor* local = local_argmax(g, x, y, kernel, stride, pad);
    return scatter_to_argmax(g, grad, local, xs, gs, kernel, stride, pad);
}

// One pool VJP for every rank: RANK is the number of pooled axes.
template <int RANK, bool IS_MAX>
class PoolVjp final : public VjpEmitter {
public:
    explicit PoolVjp(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;

        const auto* K = int_vec_attr_local(node, "kernel_size");
        const auto* S = int_vec_attr_local(node, "stride");
        const auto* P = int_vec_attr_local(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr) return false;
        if (K->size() != RANK || S->size() != RANK || P->size() != RANK) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;
        // Mixed-dtype reconciliation: the gradient kernels need grad and
        // source in one dtype (MPSGraph rejects otherwise under autocast).
        x = cast_if_needed(g, x, grad.dataType);
        NSArray<NSNumber*>* xs = x.shape;
        NSArray<NSNumber*>* gs = grad.shape;
        if (xs.count != (NSUInteger)RANK + 2 || gs.count != xs.count || !known(xs) || !known(gs))
            return false;

        const bool ceil = flag_attr(node, "ceil_mode", false);
        bool include_pad = true;
        if (!IS_MAX) {
            include_pad = flag_attr(node, "count_include_pad", true);
            if (!settle_ceil_divisor(node, ceil, include_pad))
                return false;
        }

        MPSGraphTensor* dx = nil;
        if (RANK <= 2) {
            // 1-D: (B, C, L) pooled as (B, C, 1, L).
            NSArray<NSNumber*>* x4 = RANK == 1 ? @[ xs[0], xs[1], @1, xs[2] ] : xs;
            NSArray<NSNumber*>* g4 = RANK == 1 ? @[ gs[0], gs[1], @1, gs[2] ] : gs;
            MPSGraphTensor* xl = [g reshapeTensor:x withShape:x4 name:nil];
            MPSGraphTensor* gl = [g reshapeTensor:grad withShape:g4 name:nil];
            const std::int64_t kh = RANK == 1 ? 1 : (*K)[0], kw = (*K)[RANK - 1];
            const std::int64_t sh = RANK == 1 ? 1 : (*S)[0], sw = (*S)[RANK - 1];
            const std::int64_t ph = RANK == 1 ? 0 : (*P)[0], pw = (*P)[RANK - 1];
            MPSGraphPooling2DOpDescriptor* d =
                [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kw
                                                           kernelHeight:(NSUInteger)kh
                                                              strideInX:(NSUInteger)sw
                                                              strideInY:(NSUInteger)sh
                                                         paddingStyle:MPSGraphPaddingStyleExplicit
                                                           dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
            d.paddingLeft = (NSUInteger)pw;
            d.paddingRight = (NSUInteger)pw;
            d.paddingTop = (NSUInteger)ph;
            d.paddingBottom = (NSUInteger)ph;
            d.ceilMode = ceil;
            MPSGraphTensor* dx4 = nil;
            if (IS_MAX) {
                MPSGraphTensor* y4 = pooled_value(bctx, node, grad.dataType, g4, [&] {
                    return recomputed_max_pool2d(g, x, xl, d, *K, *S, *P, gs, g4);
                });
                dx4 = max_pool_grad(g, xl, y4, gl, x4, g4, {kh, kw}, {sh, sw}, {ph, pw});
            } else {
                d.includeZeroPadToAverage = include_pad;
                dx4 = [g avgPooling2DGradientWithGradientTensor:gl
                                                   sourceTensor:xl
                                                     descriptor:d
                                                           name:nil];
            }
            dx = [g reshapeTensor:dx4 withShape:xs name:nil];
        } else {
            // 3-D: (B, C, D, H, W) pooled as (B, C, 1, D, H, W) by the 4-D op.
            NSArray<NSNumber*>* x6 = @[ xs[0], xs[1], @1, xs[2], xs[3], xs[4] ];
            NSArray<NSNumber*>* g6 = @[ gs[0], gs[1], @1, gs[2], gs[3], gs[4] ];
            MPSGraphTensor* xl = [g reshapeTensor:x withShape:x6 name:nil];
            MPSGraphTensor* gl = [g reshapeTensor:grad withShape:g6 name:nil];
            MPSGraphPooling4DOpDescriptor* d = [MPSGraphPooling4DOpDescriptor
                descriptorWithKernelSizes:@[ @1, @((*K)[0]), @((*K)[1]), @((*K)[2]) ]
                                  strides:@[ @1, @((*S)[0]), @((*S)[1]), @((*S)[2]) ]
                            dilationRates:@[ @1, @1, @1, @1 ]
                            paddingValues:@[
                                @0, @0, @((*P)[0]), @((*P)[0]), @((*P)[1]), @((*P)[1]),
                                @((*P)[2]), @((*P)[2])
                            ]
                             paddingStyle:MPSGraphPaddingStyleExplicit];
            if (d == nil) return false;
            d.ceilMode = ceil;
            MPSGraphTensor* dx6 = nil;
            if (IS_MAX) {
                const std::vector<std::int64_t> k4{1, (*K)[0], (*K)[1], (*K)[2]};
                const std::vector<std::int64_t> s4{1, (*S)[0], (*S)[1], (*S)[2]};
                const std::vector<std::int64_t> p4{0, (*P)[0], (*P)[1], (*P)[2]};
                MPSGraphTensor* y6 = pooled_value(bctx, node, grad.dataType, g6, [&] {
                    return recomputed_max_pool4d(g, x, xl, d, *K, *S, *P, gs, g6);
                });
                dx6 = max_pool_grad(g, xl, y6, gl, x6, g6, k4, s4, p4);
            } else {
                d.includeZeroPadToAverage = include_pad;
                dx6 = [g avgPooling4DGradientWithGradientTensor:gl
                                                   sourceTensor:xl
                                                     descriptor:d
                                                           name:nil];
            }
            dx = [g reshapeTensor:dx6 withShape:xs name:nil];
        }
        if (dx == nil) return false;
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }

private:
    std::string name_;
};

struct PoolVjpRegistrar {
    PoolVjpRegistrar() {
        register_vjp_emitter(std::make_unique<PoolVjp<1, true>>("max_pool1d"));
        register_vjp_emitter(std::make_unique<PoolVjp<1, false>>("avg_pool1d"));
        register_vjp_emitter(std::make_unique<PoolVjp<2, true>>("max_pool2d"));
        register_vjp_emitter(std::make_unique<PoolVjp<2, false>>("avg_pool2d"));
        register_vjp_emitter(std::make_unique<PoolVjp<3, true>>("max_pool3d"));
        register_vjp_emitter(std::make_unique<PoolVjp<3, false>>("avg_pool3d"));
    }
};

[[maybe_unused]] static const PoolVjpRegistrar g_pool_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
