// lucid/_C/compile/OpEmitters/Linear.mm
//
// MPSGraph emitters for ``nn.Linear`` forward and the fused inference
// kernels behind ``nn.FusedLinear``:
//
//     linear              y = matmul(x, W^T) + b
//     _fused_linear_relu  y = relu(matmul(x, W^T) + b)
//     _fused_linear_gelu  y = gelu_tanh(matmul(x, W^T) + b)
//
// Inputs (in :func:`NaryKernel<LinearBackward,3>::wire_autograd` order, which
// the fused bindings record too):
//
//   - ``inputs[0]`` — x   (..., in_features)
//   - ``inputs[1]`` — W   (out_features, in_features)
//   - ``inputs[2]`` — b   (out_features,)  — optional for ``linear``
//                                              (``-1`` when bias=False)
//
// Output:
//
//   - (..., out_features)
//
// The fused ops have no VJP: their bindings never produce a tensor that
// requires grad (the Python wrappers take linear + activation whenever an
// operand does), so no backward pass reaches them.
//
// The emitters are registered at process startup via a file-scope
// initialiser; :class:`MpsBuilder` finds them by op name.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"
#include "../_GeluTanh.h"
#include "../_NanPropagation.h"

namespace lucid::compile {

namespace {

// The graph tensor for operand ``id``, or nil for an external feed (``-1``)
// or an id no earlier op bound.
MPSGraphTensor* operand(const BuilderContext& ctx, TensorId id) {
    return id < 0 ? nil : (__bridge MPSGraphTensor*)ctx.resolve(id);
}

// ``matmul(x, W^T) + b`` for ``node``'s operands, or nil when one is missing.
MPSGraphTensor* linear_graph(BuilderContext& ctx, const OpNode& node, bool bias_required) {
    const std::size_t n = node.inputs.size();
    const bool arity_ok = bias_required ? n == 3 : (n == 2 || n == 3);
    if (!arity_ok || node.outputs.empty())
        return nil;
    MPSGraphTensor* x_t = operand(ctx, node.inputs[0]);
    MPSGraphTensor* W_t = operand(ctx, node.inputs[1]);
    MPSGraphTensor* b_t = n == 3 ? operand(ctx, node.inputs[2]) : nil;
    if (x_t == nil || W_t == nil || (bias_required && b_t == nil))
        return nil;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();

    // W has shape (out, in); transpose to (in, out) for matmul.
    MPSGraphTensor* W_T = [graph transposeTensor:W_t permutation:@[@1, @0] name:@"linear_W_T"];

    // matmul(x, W^T) — MPSGraph supports batched 2-D and N-D.
    MPSGraphTensor* xW = [graph matrixMultiplicationWithPrimaryTensor:x_t
                                                      secondaryTensor:W_T
                                                                 name:@"linear_xWt"];

    // Optional bias add (broadcasts over leading batch dims).
    return (b_t != nil)
               ? [graph additionWithPrimaryTensor:xW secondaryTensor:b_t name:@"linear_out"]
               : xW;
}

class LinearEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "linear"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraphTensor* y = linear_graph(ctx, node, false);
        if (y == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

class FusedLinearReluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "_fused_linear_relu"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraphTensor* pre = linear_graph(ctx, node, true);
        if (pre == nil)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* y = relu_keeping_nan(graph, pre, @"fused_linear_relu");
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

class FusedLinearGeluEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "_fused_linear_gelu"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraphTensor* pre = linear_graph(ctx, node, true);
        if (pre == nil)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* y = gelu_tanh_graph(graph, pre, @"fused_linear_gelu");
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

// File-scope registrar — runs once at .so load before any user code, so the
// emitters are available the first time :class:`MpsBuilder` looks them up.
struct LinearEmitterRegistrar {
    LinearEmitterRegistrar() {
        register_emitter(std::make_unique<LinearEmitter>());
        register_emitter(std::make_unique<FusedLinearReluEmitter>());
        register_emitter(std::make_unique<FusedLinearGeluEmitter>());
    }
};

[[maybe_unused]] static const LinearEmitterRegistrar g_linear_registrar;

}  // namespace

}  // namespace lucid::compile
