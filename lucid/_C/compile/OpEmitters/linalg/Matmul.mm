// lucid/_C/compile/OpEmitters/linalg/Matmul.mm
//
// Two-input matrix-multiplication emitter.  Pulled out of the
// elementwise/Arith.mm bucket so it lives next to the rest of the
// linear-algebra primitives (Linear / Inner / Outer / Tensordot /
// MatrixOps).
//
// Engine schema: ``matmul`` (lucid/_C/ops/bfunc/Matmul.cpp).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"
#include "../_SignedZero.h"

namespace lucid::compile {

namespace {

class MatmulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "matmul"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (graph == nil || a_t == nil || b_t == nil)
            return false;
        // Eager multiplies integer matrices; MPSGraph's matmul does not decline
        // them, it aborts the process ("'mps.matmul' op operand #0 must be
        // tensor of floating point values").  ``matrix_power`` on an integer
        // matrix traces as exactly this.
        if (!(a_t.dataType & MPSDataTypeFloatBit) || !(b_t.dataType & MPSDataTypeFloatBit))
            return false;

        // Attention value-projection workaround.  A matmul whose operand is a
        // softmax output is the ``softmax(QKᵀ) @ V`` half of attention, which
        // MPSGraph pattern-matches onto a fused-attention kernel that silently
        // miscompiles for some shapes (sequence length in [17,24], batch >= 3,
        // on macOS 26).  Emit it transposed — ``a @ b == (bᵀ @ aᵀ)ᵀ`` — so the
        // matmul's direct operands are transposes, not the raw softmax output,
        // and the buggy pass does not fire.  Mathematically identical; the only
        // cost is two metadata transposes, and only on attention matmuls.
        const NSUInteger nda = a_t.shape.count;
        const NSUInteger ndb = b_t.shape.count;
        if (apply_attention_workaround() &&
            (ctx.is_softmax_output(a_id) || ctx.is_softmax_output(b_id)) && nda >= 2 && ndb >= 2) {
            MPSGraphTensor* a_tr = [graph transposeTensor:a_t
                                                dimension:(NSInteger)(nda - 1)
                                            withDimension:(NSInteger)(nda - 2)
                                                     name:nil];
            MPSGraphTensor* b_tr = [graph transposeTensor:b_t
                                                dimension:(NSInteger)(ndb - 1)
                                            withDimension:(NSInteger)(ndb - 2)
                                                     name:nil];
            MPSGraphTensor* prod = [graph matrixMultiplicationWithPrimaryTensor:b_tr
                                                                secondaryTensor:a_tr
                                                                           name:@"matmul_attn"];
            const NSUInteger ndp = prod.shape.count;
            MPSGraphTensor* y = [graph transposeTensor:prod
                                             dimension:(NSInteger)(ndp - 1)
                                         withDimension:(NSInteger)(ndp - 2)
                                                  name:@"matmul"];
            ctx.bind(node.outputs[0].id, (__bridge void*)y);
            return true;
        }

        a_t = materialize_matmul_splat(graph, a_t, is_const_splat(ctx, a_id));
        b_t = materialize_matmul_splat(graph, b_t, is_const_splat(ctx, b_id));

        MPSGraphTensor* y = [graph matrixMultiplicationWithPrimaryTensor:a_t
                                                         secondaryTensor:b_t
                                                                    name:@"matmul"];
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

struct MatmulEmitterRegistrar {
    MatmulEmitterRegistrar() { register_emitter(std::make_unique<MatmulEmitter>()); }
};

[[maybe_unused]] static const MatmulEmitterRegistrar g_matmul_registrar;

}  // namespace

}  // namespace lucid::compile
