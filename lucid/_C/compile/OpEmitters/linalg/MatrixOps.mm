// lucid/_C/compile/OpEmitters/linalg/MatrixOps.mm
//
// Linear-algebra emitters that translate 1:1 (or with a small
// MPSGraph decomposition) to a graph operation:
//
//   - ``norm``           — vector / matrix p-norm with ord=1/2/inf fast paths
//   - ``matrix_power``   — repeated matmul via binary exponentiation
//   - ``det``            — 2×2 / 3×3 closed-form (eager fallback for N>3)
//   - ``inv``            — 2×2 closed-form via adjugate / det
//   - ``tensordot``      — permute + reshape + matmul
//   - ``inner``          — last-axis contraction: reshape + matmul with bᵀ
//   - ``outer``          — broadcast multiply of two 1-D vectors
//   - ``dot``            — element-wise multiply + full reduce-sum
//   - ``trace``          — sum of the main diagonal (2-D only)
//   - ``bilinear_layer`` — y[d] = x1 @ W[d] @ x2ᵀ + b[d] for each output d
//
// Heavier linalg ops (cholesky / qr / svd / eig / eigh / solve / pinv)
// remain stubs in :file:`../misc/Stubs.mm` because they require
// iterative algorithms MPSGraph does not provide natively.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// ── norm — sum(|x|^ord, axes)^(1/ord) with ord=1/2/inf fast paths.
class NormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        double ord = double_attr(node, "ord", 2.0);
        bool keepdims = bool_attr(node, "keepdims", false);
        // Pull axis list — if absent or empty, reduce over all axes.
        NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
        if (const auto* v = int_vec_attr(node, "axis_list")) {
            NSInteger nd = (NSInteger)x.shape.count;
            for (auto a : *v) {
                NSInteger p = (NSInteger)a;
                if (p < 0) p += nd;
                [axes addObject:[NSNumber numberWithLongLong:p]];
            }
        }
        if (axes.count == 0) {
            for (NSUInteger d = 0; d < x.shape.count; ++d)
                [axes addObject:[NSNumber numberWithLongLong:(long long)d]];
        }
        MPSGraphTensor* abs_x = [g absoluteWithTensor:x name:nil];
        MPSGraphTensor* r;
        if (ord == 1.0) {
            r = [g reductionSumWithTensor:abs_x axes:axes name:nil];
        } else if (ord == 2.0) {
            MPSGraphTensor* sq = [g squareWithTensor:x name:nil];
            MPSGraphTensor* s = [g reductionSumWithTensor:sq axes:axes name:nil];
            r = [g squareRootWithTensor:s name:nil];
        } else if (std::isinf(ord) && ord > 0) {
            r = [g reductionMaximumWithTensor:abs_x axes:axes name:nil];
        } else if (std::isinf(ord) && ord < 0) {
            r = [g reductionMinimumWithTensor:abs_x axes:axes name:nil];
        } else {
            MPSGraphTensor* ord_c = [g constantWithScalar:ord dataType:x.dataType];
            MPSGraphTensor* powed =
                [g powerWithPrimaryTensor:abs_x secondaryTensor:ord_c name:nil];
            MPSGraphTensor* s = [g reductionSumWithTensor:powed axes:axes name:nil];
            MPSGraphTensor* inv_ord_c =
                [g constantWithScalar:(1.0 / ord) dataType:x.dataType];
            r = [g powerWithPrimaryTensor:s secondaryTensor:inv_ord_c name:nil];
        }
        // Re-insert reduced axes as size-1 dims when keepdims=true.
        if (keepdims) {
            NSMutableArray<NSNumber*>* kept = [NSMutableArray array];
            std::vector<bool> mask(x.shape.count, false);
            for (NSNumber* a in axes) {
                NSInteger p = a.longLongValue;
                if (p >= 0 && p < (NSInteger)x.shape.count) mask[p] = true;
            }
            for (NSUInteger d = 0; d < x.shape.count; ++d) {
                if (mask[d]) [kept addObject:@1];
                else [kept addObject:x.shape[d]];
            }
            r = [g reshapeTensor:r withShape:kept name:nil];
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(r));
        return true;
    }
};

// ── matrix_power — repeated matmul via binary exponentiation, p ≥ 0.
class MatrixPowerEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "matrix_power"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t p = int_attr(node, "p", 1);
        if (p < 0) return false;  // negative power needs inv — fall back to eager
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSArray<NSNumber*>* sh = x.shape;
        if (sh.count != 2) return false;
        std::int64_t N = sh[0].longLongValue;
        if (sh[1].longLongValue != N) return false;
        if (p == 0) {
            // Identity I_N: broadcast ones, mask with bandPart(0, 0).
            MPSGraphTensor* ones_full = [g constantWithScalar:1.0 dataType:x.dataType];
            ones_full = [g broadcastTensor:ones_full toShape:@[sh[0], sh[1]] name:nil];
            ctx.bind(node.outputs[0].id, (__bridge void*)([g bandPartWithTensor:ones_full
                                                numLower:0
                                                numUpper:0
                                                    name:@"matpow_eye"]));
        return true;
        }
        // Eager computes an integer matrix power; MPSGraph's matmul does not
        // decline an integer operand, it aborts the process
        // ("'mps.matmul' op operand #0 must be tensor of floating point").
        if (p >= 2 && !(x.dataType & MPSDataTypeFloatBit))
            return false;
        MPSGraphTensor* result = nil;
        MPSGraphTensor* base = x;
        std::int64_t e = p;
        while (e > 0) {
            if (e & 1) {
                result = result == nil
                             ? base
                             : [g matrixMultiplicationWithPrimaryTensor:result
                                                         secondaryTensor:base
                                                                    name:nil];
            }
            e >>= 1;
            if (e > 0) {
                base = [g matrixMultiplicationWithPrimaryTensor:base
                                                  secondaryTensor:base
                                                             name:nil];
            }
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(result));
        return true;
    }
};

// ── det — 2×2 / 3×3 closed-form, eager fallback for N>3.
class DetEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "det"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSArray<NSNumber*>* sh = x.shape;
        if (sh.count != 2) return false;
        std::int64_t N = sh[0].longLongValue;
        if (sh[1].longLongValue != N) return false;
        auto el = [&](NSInteger r, NSInteger c) -> MPSGraphTensor* {
            MPSGraphTensor* row =
                [g sliceTensor:x dimension:0 start:r length:1 name:nil];
            MPSGraphTensor* cel =
                [g sliceTensor:row dimension:1 start:c length:1 name:nil];
            return [g reshapeTensor:cel withShape:@[] name:nil];
        };
        if (N == 1) {
            ctx.bind(node.outputs[0].id,
                     (__bridge void*)([g reshapeTensor:x withShape:@[] name:@"det1"]));
            return true;
        }
        if (N == 2) {
            MPSGraphTensor* a = el(0, 0), *b = el(0, 1);
            MPSGraphTensor* c = el(1, 0), *d = el(1, 1);
            MPSGraphTensor* ad = [g multiplicationWithPrimaryTensor:a secondaryTensor:d name:nil];
            MPSGraphTensor* bc = [g multiplicationWithPrimaryTensor:b secondaryTensor:c name:nil];
            ctx.bind(node.outputs[0].id, (__bridge void*)([g subtractionWithPrimaryTensor:ad
                                                    secondaryTensor:bc
                                                               name:@"det2"]));
        return true;
        }
        if (N == 3) {
            MPSGraphTensor* a00 = el(0, 0), *a01 = el(0, 1), *a02 = el(0, 2);
            MPSGraphTensor* a10 = el(1, 0), *a11 = el(1, 1), *a12 = el(1, 2);
            MPSGraphTensor* a20 = el(2, 0), *a21 = el(2, 1), *a22 = el(2, 2);
            auto sub_pair = ^MPSGraphTensor*(MPSGraphTensor* p1, MPSGraphTensor* p2,
                                              MPSGraphTensor* p3, MPSGraphTensor* p4) {
                MPSGraphTensor* m1 =
                    [g multiplicationWithPrimaryTensor:p1 secondaryTensor:p2 name:nil];
                MPSGraphTensor* m2 =
                    [g multiplicationWithPrimaryTensor:p3 secondaryTensor:p4 name:nil];
                return [g subtractionWithPrimaryTensor:m1 secondaryTensor:m2 name:nil];
            };
            MPSGraphTensor* cof0 = sub_pair(a11, a22, a12, a21);
            MPSGraphTensor* cof1 = sub_pair(a10, a22, a12, a20);
            MPSGraphTensor* cof2 = sub_pair(a10, a21, a11, a20);
            MPSGraphTensor* t0 =
                [g multiplicationWithPrimaryTensor:a00 secondaryTensor:cof0 name:nil];
            MPSGraphTensor* t1 =
                [g multiplicationWithPrimaryTensor:a01 secondaryTensor:cof1 name:nil];
            MPSGraphTensor* t2 =
                [g multiplicationWithPrimaryTensor:a02 secondaryTensor:cof2 name:nil];
            MPSGraphTensor* d01 = [g subtractionWithPrimaryTensor:t0 secondaryTensor:t1 name:nil];
            ctx.bind(node.outputs[0].id, (__bridge void*)([g additionWithPrimaryTensor:d01
                                                  secondaryTensor:t2
                                                             name:@"det3"]));
        return true;
        }
        return false;  // N > 3 — fall back to eager LAPACK.
    }
};

// ── inv — 2×2 closed-form via adjugate / det.  3×3 is too verbose
// and N > 3 needs LU; both fall through to nullptr → eager.
class InvEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "inv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSArray<NSNumber*>* sh = x.shape;
        if (sh.count != 2) return false;
        std::int64_t N = sh[0].longLongValue;
        if (sh[1].longLongValue != N) return false;
        if (N == 1) {
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
            ctx.bind(node.outputs[0].id,
                     (__bridge void*)([g divisionWithPrimaryTensor:one
                                                   secondaryTensor:x
                                                              name:@"inv1"]));
            return true;
        }
        if (N != 2) return false;
        auto el = [&](NSInteger r, NSInteger c) -> MPSGraphTensor* {
            MPSGraphTensor* row =
                [g sliceTensor:x dimension:0 start:r length:1 name:nil];
            MPSGraphTensor* cel =
                [g sliceTensor:row dimension:1 start:c length:1 name:nil];
            return [g reshapeTensor:cel withShape:@[] name:nil];
        };
        MPSGraphTensor* a = el(0, 0), *b = el(0, 1);
        MPSGraphTensor* c = el(1, 0), *d = el(1, 1);
        MPSGraphTensor* ad = [g multiplicationWithPrimaryTensor:a secondaryTensor:d name:nil];
        MPSGraphTensor* bc = [g multiplicationWithPrimaryTensor:b secondaryTensor:c name:nil];
        MPSGraphTensor* det = [g subtractionWithPrimaryTensor:ad secondaryTensor:bc name:nil];
        MPSGraphTensor* neg_b = [g negativeWithTensor:b name:nil];
        MPSGraphTensor* neg_c = [g negativeWithTensor:c name:nil];
        // Assemble adj = [[d, -b], [-c, a]] / det.
        NSArray<NSNumber*>* one_one = @[@1, @1];
        MPSGraphTensor* row0_l = [g reshapeTensor:d withShape:one_one name:nil];
        MPSGraphTensor* row0_r = [g reshapeTensor:neg_b withShape:one_one name:nil];
        MPSGraphTensor* row1_l = [g reshapeTensor:neg_c withShape:one_one name:nil];
        MPSGraphTensor* row1_r = [g reshapeTensor:a withShape:one_one name:nil];
        MPSGraphTensor* row0 =
            [g concatTensors:@[row0_l, row0_r] dimension:1 name:nil];
        MPSGraphTensor* row1 =
            [g concatTensors:@[row1_l, row1_r] dimension:1 name:nil];
        MPSGraphTensor* adj = [g concatTensors:@[row0, row1] dimension:0 name:nil];
        MPSGraphTensor* det_b = [g reshapeTensor:det withShape:one_one name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g divisionWithPrimaryTensor:adj
                                             secondaryTensor:det_b
                                                        name:@"inv2"]));
        return true;
    }
};

// ── tensordot — permute contracted axes to the boundary, flatten, matmul.
class TensordotEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "tensordot"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty()) return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0) return false;
        std::vector<std::int64_t> ax_a, ax_b;
        if (const auto* v = int_vec_attr(node, "axes_a")) ax_a = *v;
        if (const auto* v = int_vec_attr(node, "axes_b")) ax_b = *v;
        if (ax_a.size() != ax_b.size() || ax_a.empty()) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || a == nil || b == nil) return false;
        // Same reason as ``MatmulEmitter``: an integer operand aborts the
        // process inside MPSGraph's matmul instead of declining.
        if (!(a.dataType & MPSDataTypeFloatBit) || !(b.dataType & MPSDataTypeFloatBit)) return false;
        NSInteger na = (NSInteger)a.shape.count;
        NSInteger nb = (NSInteger)b.shape.count;
        std::vector<bool> a_is_c(na, false), b_is_c(nb, false);
        for (std::size_t i = 0; i < ax_a.size(); ++i) {
            std::int64_t p = ax_a[i]; if (p < 0) p += na;
            std::int64_t q = ax_b[i]; if (q < 0) q += nb;
            if (p < 0 || p >= na || q < 0 || q >= nb) return false;
            a_is_c[p] = true; b_is_c[q] = true;
        }
        // A: free dims first, then contracted (in input order).
        NSMutableArray<NSNumber*>* a_perm = [NSMutableArray array];
        std::vector<NSNumber*> a_kept;
        for (NSInteger d = 0; d < na; ++d)
            if (!a_is_c[d]) {
                [a_perm addObject:[NSNumber numberWithLongLong:d]];
                a_kept.push_back(a.shape[d]);
            }
        for (std::size_t i = 0; i < ax_a.size(); ++i) {
            std::int64_t p = ax_a[i]; if (p < 0) p += na;
            [a_perm addObject:[NSNumber numberWithLongLong:p]];
        }
        // B: contracted first (matching A's order), then free.
        NSMutableArray<NSNumber*>* b_perm = [NSMutableArray array];
        for (std::size_t i = 0; i < ax_b.size(); ++i) {
            std::int64_t q = ax_b[i]; if (q < 0) q += nb;
            [b_perm addObject:[NSNumber numberWithLongLong:q]];
        }
        std::vector<NSNumber*> b_kept;
        for (NSInteger d = 0; d < nb; ++d)
            if (!b_is_c[d]) {
                [b_perm addObject:[NSNumber numberWithLongLong:d]];
                b_kept.push_back(b.shape[d]);
            }
        MPSGraphTensor* a_p = [g transposeTensor:a permutation:a_perm name:nil];
        MPSGraphTensor* b_p = [g transposeTensor:b permutation:b_perm name:nil];
        std::int64_t M = 1, K = 1, N = 1;
        for (NSNumber* n : a_kept) M *= n.longLongValue;
        for (NSNumber* n : b_kept) N *= n.longLongValue;
        for (std::size_t i = 0; i < ax_a.size(); ++i) {
            std::int64_t p = ax_a[i]; if (p < 0) p += na;
            K *= a.shape[p].longLongValue;
        }
        // A symbolic batch axis (-1) among the flattened ones makes a
        // product like -64, which the reshape below would take as a size.
        // Decline so the builder retries with static shapes.
        for (NSNumber* n in a.shape) if (n.longLongValue < 0) return false;
        for (NSNumber* n in b.shape) if (n.longLongValue < 0) return false;
        a_p = [g reshapeTensor:a_p
                     withShape:@[[NSNumber numberWithLongLong:M],
                                  [NSNumber numberWithLongLong:K]]
                          name:nil];
        b_p = [g reshapeTensor:b_p
                     withShape:@[[NSNumber numberWithLongLong:K],
                                  [NSNumber numberWithLongLong:N]]
                          name:nil];
        MPSGraphTensor* c =
            [g matrixMultiplicationWithPrimaryTensor:a_p secondaryTensor:b_p name:@"tensordot"];
        NSMutableArray<NSNumber*>* out_sh = [NSMutableArray array];
        for (NSNumber* n : a_kept) [out_sh addObject:n];
        for (NSNumber* n : b_kept) [out_sh addObject:n];
        // ``out_sh`` is empty for a full contraction, which reshapes to a
        // scalar — one call covers both.  This used to bind only that case
        // and return before the general one, so every non-scalar
        // tensordot left its output unbound and the consumer declined.
        ctx.bind(node.outputs[0].id, (__bridge void*)([g reshapeTensor:c withShape:out_sh name:nil]));
        return true;
    }
};

// ── inner — contract the last axis of each operand:
//    a[..., K] · b[..., K] → a.shape[:-1] + b.shape[:-1].
//    Flattened, that is a[M, K] @ b[N, K]ᵀ.  An element-wise product
//    summed over the last axis is the same thing only for two vectors —
//    this emitter used to do exactly that, and gave [M] for [M, K] ·
//    [M, K] where the answer is [M, M].
class InnerEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "inner"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty()) return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || a == nil || b == nil) return false;
        // MPSGraph's matmul takes floating point only and aborts the
        // process on anything else; eager ``inner`` rejects integers too.
        if (!(a.dataType & MPSDataTypeFloatBit) || !(b.dataType & MPSDataTypeFloatBit)) return false;
        NSArray<NSNumber*>* sa = a.shape;
        NSArray<NSNumber*>* sb = b.shape;
        if (sa.count == 0 || sb.count == 0) return false;
        const long long K = sa.lastObject.longLongValue;
        if (K < 0 || sb.lastObject.longLongValue != K) return false;
        long long M = 1, N = 1;
        for (NSUInteger d = 0; d + 1 < sa.count; ++d) M *= sa[d].longLongValue;
        for (NSUInteger d = 0; d + 1 < sb.count; ++d) N *= sb[d].longLongValue;
        if (M < 0 || N < 0) return false;  // symbolic batch — the flatten would pin it
        MPSGraphTensor* a2 = [g reshapeTensor:a withShape:@[@(M), @(K)] name:nil];
        MPSGraphTensor* b2 = [g reshapeTensor:b withShape:@[@(N), @(K)] name:nil];
        MPSGraphTensor* bt = [g transposeTensor:b2 dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* c = [g matrixMultiplicationWithPrimaryTensor:a2 secondaryTensor:bt name:nil];
        NSMutableArray<NSNumber*>* out_sh = [NSMutableArray array];
        for (std::int64_t d : node.outputs[0].shape) [out_sh addObject:@(d)];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g reshapeTensor:c withShape:out_sh name:@"inner"]));
        return true;
    }
};

// ── outer — broadcast multiply of two 1-D vectors.
class OuterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "outer"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty()) return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || a == nil || b == nil) return false;
        if (a.shape.count != 1 || b.shape.count != 1) return false;
        MPSGraphTensor* a2 = [g reshapeTensor:a withShape:@[a.shape[0], @1] name:nil];
        MPSGraphTensor* b2 = [g reshapeTensor:b withShape:@[@1, b.shape[0]] name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g multiplicationWithPrimaryTensor:a2
                                                   secondaryTensor:b2
                                                              name:@"outer"]));
        return true;
    }
};

// ── dot (1-D) — element-wise multiply + full reduce-sum to scalar.
class DotEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "dot"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty()) return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || a == nil || b == nil) return false;
        MPSGraphTensor* prod =
            [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
        NSUInteger nd = a.shape.count;
        NSMutableArray<NSNumber*>* all_axes = [NSMutableArray arrayWithCapacity:nd];
        for (NSUInteger d = 0; d < nd; ++d)
            [all_axes addObject:[NSNumber numberWithLongLong:(long long)d]];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g reductionSumWithTensor:prod axes:all_axes name:@"dot"]));
        return true;
    }
};

// ── trace — sum of main diagonal of a 2-D tensor.
class TraceEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "trace"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        if (x.shape.count != 2) return false;
        MPSGraphTensor* band =
            [g bandPartWithTensor:x numLower:0 numUpper:0 name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g reductionSumWithTensor:band
                                                     axes:@[@0, @1]
                                                     name:@"trace"]));
        return true;
    }
};

// ── bilinear_layer — y[d] = x1 @ W[d] @ x2ᵀ + b[d].
// Algorithm: permute W to (D1, Dout, D2), reshape to (D1, Dout*D2), then
//   t1_flat = x1 (B, D1) @ Wflat (D1, Dout*D2)        → (B, Dout*D2)
//   t1      = reshape(t1_flat, (B, Dout, D2))
//   y       = sum(t1 * x2.unsqueeze(1), axis=-1)       → (B, Dout)
// Bias (Dout,) is broadcast-added at the end if present.
class BilinearLayerEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bilinear_layer"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty()) return false;
        TensorId x1_id = node.inputs[0];
        TensorId x2_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        if (x1_id < 0 || x2_id < 0 || w_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x1 = (__bridge MPSGraphTensor*)ctx.resolve(x1_id);
        MPSGraphTensor* x2 = (__bridge MPSGraphTensor*)ctx.resolve(x2_id);
        MPSGraphTensor* W = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        if (g == nil || x1 == nil || x2 == nil || W == nil) return false;
        if (x1.shape.count != 2 || x2.shape.count != 2 || W.shape.count != 3)
            return false;
        NSNumber* B = x1.shape[0];
        NSNumber* D1 = x1.shape[1];
        NSNumber* D2 = x2.shape[1];
        NSNumber* Dout = W.shape[0];
        // Decomposition: y[b, d] = Σ_{i,j} x1[b, i] · W[d, i, j] · x2[b, j].
        // 1. outer[b, i, j] = x1[b, i] · x2[b, j]                — (B, D1, D2)
        // 2. outer_flat[b, k] = outer[b, i*D2 + j]               — (B, D1·D2)
        // 3. W_flat[d, k] = W[d, i*D2 + j]                       — (Dout, D1·D2)
        // 4. y = outer_flat @ W_flat.T                            — (B, Dout)
        const std::int64_t K = D1.longLongValue * D2.longLongValue;
        NSArray<NSNumber*>* x1_e_sh = @[B, D1, @1];
        NSArray<NSNumber*>* x2_e_sh = @[B, @1, D2];
        MPSGraphTensor* x1_e = [g reshapeTensor:x1 withShape:x1_e_sh name:nil];
        MPSGraphTensor* x2_e = [g reshapeTensor:x2 withShape:x2_e_sh name:nil];
        MPSGraphTensor* outer =
            [g multiplicationWithPrimaryTensor:x1_e secondaryTensor:x2_e name:nil];
        NSArray<NSNumber*>* outer_flat_sh =
            @[B, [NSNumber numberWithLongLong:K]];
        MPSGraphTensor* outer_flat =
            [g reshapeTensor:outer withShape:outer_flat_sh name:nil];
        NSArray<NSNumber*>* W_flat_sh =
            @[Dout, [NSNumber numberWithLongLong:K]];
        MPSGraphTensor* W_flat = [g reshapeTensor:W withShape:W_flat_sh name:nil];
        // W_flat^T: (K, Dout) via swap dim 0 ↔ 1
        MPSGraphTensor* W_flat_T =
            [g transposeTensor:W_flat dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* y =
            [g matrixMultiplicationWithPrimaryTensor:outer_flat
                                       secondaryTensor:W_flat_T
                                                  name:nil];
        if (node.inputs.size() >= 4) {
            TensorId b_id = node.inputs[3];
            if (b_id >= 0) {
                MPSGraphTensor* bias = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
                if (bias != nil) {
                    MPSGraphTensor* b_r =
                        [g reshapeTensor:bias withShape:@[@1, Dout] name:nil];
                    y = [g additionWithPrimaryTensor:y secondaryTensor:b_r name:nil];
                }
            }
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

struct MatrixOpsRegistrar {
    MatrixOpsRegistrar() {
        register_emitter(std::make_unique<NormEmitter>());
        register_emitter(std::make_unique<MatrixPowerEmitter>());
        register_emitter(std::make_unique<DetEmitter>());
        register_emitter(std::make_unique<InvEmitter>());
        register_emitter(std::make_unique<TensordotEmitter>());
        register_emitter(std::make_unique<InnerEmitter>());
        register_emitter(std::make_unique<OuterEmitter>());
        register_emitter(std::make_unique<DotEmitter>());
        register_emitter(std::make_unique<TraceEmitter>());
        register_emitter(std::make_unique<BilinearLayerEmitter>());
    }
};

[[maybe_unused]] static const MatrixOpsRegistrar g_matrix_ops_registrar;

}  // namespace

}  // namespace lucid::compile
