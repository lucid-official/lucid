// lucid/_C/compile/OpEmitters/Pool.mm
//
// MaxPool2d / AvgPool2d emitters.
//
// Op names match the engine schemas in lucid/_C/nn/PoolNd.cpp
// ("max_pool2d", "avg_pool2d").  Each carries vector<int64> attrs
// ``kernel_size`` / ``stride`` / ``padding`` reported by the forward
// via :func:`OpScopeFull::set_attr`.
//
// Lucid's tensors are NCHW; MPSGraph's 2-D pool descriptors default to
// NHWC, so the descriptor's ``dataLayout`` is set to
// ``NCHW`` explicitly.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"
#include "../_AttrHelpers.h"
#include "../_MaxPool.h"

namespace lucid::compile {

namespace {

class MaxPool2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "max_pool2d"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr)
            return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;

        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[1];
        d.paddingRight = (NSUInteger)(*P)[1];
        d.paddingTop = (NSUInteger)(*P)[0];
        d.paddingBottom = (NSUInteger)(*P)[0];
        d.ceilMode = flag_attr(node, "ceil_mode", false);

        MPSGraphTensor* y =
            [graph maxPooling2DWithSourceTensor:x_t descriptor:d name:@"max_pool2d"];
        if (!matches_recorded_shape(y, node))
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

class AvgPool2dEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "avg_pool2d"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr)
            return false;
        if (K->size() != 2 || S->size() != 2 || P->size() != 2)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;

        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[1]
                                                       kernelHeight:(NSUInteger)(*K)[0]
                                                          strideInX:(NSUInteger)(*S)[1]
                                                          strideInY:(NSUInteger)(*S)[0]
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[1];
        d.paddingRight = (NSUInteger)(*P)[1];
        d.paddingTop = (NSUInteger)(*P)[0];
        d.paddingBottom = (NSUInteger)(*P)[0];
        // The divisor follows the recorded ``count_include_pad`` (default
        // True, as in the reference); it used to be hard-wired to True, so
        // ``count_include_pad=False`` averaged the padding in anyway.
        d.includeZeroPadToAverage = flag_attr(node, "count_include_pad", true);
        d.ceilMode = flag_attr(node, "ceil_mode", false);
        // Ceil mode with the padding counted — see settle_ceil_divisor.
        bool include_pad = d.includeZeroPadToAverage;
        if (!settle_ceil_divisor(node, d.ceilMode, include_pad))
            return false;
        d.includeZeroPadToAverage = include_pad;

        MPSGraphTensor* y =
            [graph avgPooling2DWithSourceTensor:x_t descriptor:d name:@"avg_pool2d"];
        if (!matches_recorded_shape(y, node))
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// ── MaxPool1d — reshape (B,C,L)→(B,C,1,L), maxPool2D, reshape back.
template <bool IS_MAX>
class Pool1dEmitterT final : public OpEmitter {
public:
    explicit Pool1dEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr) return false;
        if (K->size() != 1 || S->size() != 1 || P->size() != 1) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        if (x.shape.count != 3) return false;
        NSArray<NSNumber*>* x4 = @[x.shape[0], x.shape[1], @1, x.shape[2]];
        MPSGraphTensor* x_r = [g reshapeTensor:x withShape:x4 name:nil];
        MPSGraphPooling2DOpDescriptor* d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)(*K)[0]
                                                       kernelHeight:1
                                                          strideInX:(NSUInteger)(*S)[0]
                                                          strideInY:1
                                                     paddingStyle:MPSGraphPaddingStyleExplicit
                                                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        d.paddingLeft = (NSUInteger)(*P)[0];
        d.paddingRight = (NSUInteger)(*P)[0];
        d.paddingTop = 0;
        d.paddingBottom = 0;
        d.ceilMode = flag_attr(node, "ceil_mode", false);
        MPSGraphTensor* y4;
        if (IS_MAX) {
            y4 = [g maxPooling2DWithSourceTensor:x_r descriptor:d name:@"max_pool1d_lifted"];
        } else {
            bool include_pad = flag_attr(node, "count_include_pad", true);
            if (!settle_ceil_divisor(node, d.ceilMode, include_pad))
                return false;
            d.includeZeroPadToAverage = include_pad;
            y4 = [g avgPooling2DWithSourceTensor:x_r descriptor:d name:@"avg_pool1d_lifted"];
        }
        // Squeeze H=1.
        NSArray<NSNumber*>* out_sh = @[y4.shape[0], y4.shape[1], y4.shape[3]];
        MPSGraphTensor* y = [g reshapeTensor:y4 withShape:out_sh name:nil];
        if (!matches_recorded_shape(y, node))
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }

private:
    std::string name_;
};

// ── 3-D pooling, through the SDK's 4-D operation.
//
// MPSGraph ships 2-D and 4-D pooling and nothing between, and Lucid's
// 3-D pool is a rank-5 tensor — which is why this was a stub. The gap is
// a reshape: a length-1 spatial axis in front makes the volume rank 6,
// the 4-D operation pools it with a kernel of 1 on that axis, and the
// axis comes back out. The descriptor takes four of everything, so every
// list is the 3-D one with a leading identity entry.
template <bool IsMax>
class Pool3dEmitterT final : public OpEmitter {
public:
    explicit Pool3dEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        if (K == nullptr || S == nullptr || P == nullptr)
            return false;
        if (K->size() != 3 || S->size() != 3 || P->size() != 3)
            return false;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return false;
        NSArray<NSNumber*>* sh = x.shape;
        if (sh.count != 5)
            return false;

        MPSGraphTensor* lifted =
            [g reshapeTensor:x
                   withShape:@[ sh[0], sh[1], @1, sh[2], sh[3], sh[4] ]
                        name:nil];

        MPSGraphPooling4DOpDescriptor* d = [MPSGraphPooling4DOpDescriptor
            descriptorWithKernelSizes:@[ @1, @((*K)[0]), @((*K)[1]), @((*K)[2]) ]
                              strides:@[ @1, @((*S)[0]), @((*S)[1]), @((*S)[2]) ]
                        dilationRates:@[ @1, @1, @1, @1 ]
                        paddingValues:@[
                            @0, @0, @((*P)[0]), @((*P)[0]), @((*P)[1]), @((*P)[1]),
                            @((*P)[2]), @((*P)[2])
                        ]
                         paddingStyle:MPSGraphPaddingStyleExplicit];
        if (d == nil)
            return false;
        d.ceilMode = flag_attr(node, "ceil_mode", false);
        if (!IsMax) {
            // Lucid's ``count_include_pad`` and MPSGraph's
            // ``includeZeroPadToAverage`` mean the same thing — except at a
            // ceil-mode overhang (see settle_ceil_divisor).
            bool include_pad = flag_attr(node, "count_include_pad", true);
            if (!settle_ceil_divisor(node, d.ceilMode, include_pad))
                return false;
            d.includeZeroPadToAverage = include_pad;
        }

        MPSGraphTensor* pooled =
            IsMax ? [g maxPooling4DWithSourceTensor:lifted descriptor:d name:nil]
                  : [g avgPooling4DWithSourceTensor:lifted descriptor:d name:nil];
        if (pooled == nil)
            return false;

        // Drop the axis that was only there to reach the 4-D operation.
        NSArray<NSNumber*>* out = pooled.shape;
        if (out.count != 6)
            return false;
        MPSGraphTensor* y = [g reshapeTensor:pooled
                                   withShape:@[ out[0], out[1], out[3], out[4], out[5] ]
                                        name:@"pool3d"];
        if (!matches_recorded_shape(y, node))
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }

private:
    std::string name_;
};


// A max pool that keeps an all -inf window at -inf (LCD-916; the plain pool answers 0 there):
// tries the stencil form first and hands the node to ``plain`` when it does not apply.
class ExactMaxPoolEmitter final : public OpEmitter {
public:
    ExactMaxPoolEmitter(std::unique_ptr<OpEmitter> plain, std::size_t nd)
        : plain_(std::move(plain)), nd_(nd) {}
    std::string_view op_name() const override { return plain_->op_name(); }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const auto* K = int_vec_attr(node, "kernel_size");
        const auto* S = int_vec_attr(node, "stride");
        const auto* P = int_vec_attr(node, "padding");
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        if (node.inputs.size() != 1 || node.inputs[0] < 0 || K == nullptr || S == nullptr ||
            P == nullptr || g == nil || node.outputs.empty())
            return plain_->emit(ctx, node);
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        const auto& shape = node.outputs[0].shape;
        if (x == nil || shape.size() < nd_)
            return plain_->emit(ctx, node);
        const std::vector<std::int64_t> pooled(shape.end() - static_cast<std::ptrdiff_t>(nd_),
                                               shape.end());
        MPSGraphTensor* y = exact_max_pool(g, x, *K, *S, *P, pooled);
        if (y == nil || !matches_recorded_shape(y, node))
            return plain_->emit(ctx, node);
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }

private:
    std::unique_ptr<OpEmitter> plain_;
    std::size_t nd_;
};

struct PoolEmitterRegistrar {
    PoolEmitterRegistrar() {
        register_emitter(std::make_unique<ExactMaxPoolEmitter>(
            std::make_unique<MaxPool2dEmitter>(), 2));
        register_emitter(std::make_unique<AvgPool2dEmitter>());
        register_emitter(std::make_unique<ExactMaxPoolEmitter>(
            std::make_unique<Pool1dEmitterT<true>>("max_pool1d"), 1));
        register_emitter(std::make_unique<Pool1dEmitterT<false>>("avg_pool1d"));
        register_emitter(std::make_unique<ExactMaxPoolEmitter>(
            std::make_unique<Pool3dEmitterT<true>>("max_pool3d"), 3));
        register_emitter(std::make_unique<Pool3dEmitterT<false>>("avg_pool3d"));
    }
};

[[maybe_unused]] static const PoolEmitterRegistrar g_pool_registrar;

}  // namespace

}  // namespace lucid::compile
