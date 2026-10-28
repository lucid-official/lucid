// lucid/_C/compile/VjpEmitters/linalg/Products.mm
//
// VJPs for the products and matrix functions that had none: ``dot``,
// ``inner``, ``outer``, ``tensordot``, ``bilinear_layer``, ``trace``,
// ``diagonal``, ``inv`` and ``det``.  Without them a training step through
// any of these depended on MPSGraph's autodiff, which a train-mode batch
// norm or an interpolation elsewhere in the graph rules out.
//
//   dot(a, b)            1-D:  da = g·b, db = g·a
//   inner(a, b) = a bᵀ   2-D:  da = G b, db = Gᵀ a
//   outer(a, b) = a bᵀ   1-D:  da = G b, db = Gᵀ a
//   tensordot            permute + flatten to one matmul, as the forward
//                        does; the matmul's VJP; unflatten + inverse permute
//   bilinear(x1, x2, W)  y[n,o] = Σ x1[n,i] W[o,i,j] x2[n,j]
//   trace(A)             dA = g · I
//   diagonal(A, k)       dA = g placed back on the k-th diagonal
//   inv(A) = Y           dA = −Yᵀ G Yᵀ
//   det(A)               dA = g · cof(A), for the 2×2 / 3×3 the forward
//                        lowers (a larger det runs eager already)

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../../OpEmitters/_AttrHelpers.h"
#include "../../OpEmitters/linalg/Decompositions.h"
#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

MPSGraphTensor* mm(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
}

MPSGraphTensor* tr(MPSGraph* g, MPSGraphTensor* t) {
    const NSInteger n = (NSInteger)t.shape.count;
    return [g transposeTensor:t dimension:n - 1 withDimension:n - 2 name:nil];
}

bool known(MPSGraphTensor* t) {
    for (NSNumber* d in t.shape)
        if (d.longLongValue < 0)
            return false;
    return true;
}

// Resolve the node's inputs and grad, cast to the grad's dtype.
struct Ctx {
    MPSGraph* g = nil;
    MPSGraphTensor* go = nil;
    std::vector<MPSGraphTensor*> in;
};

bool open(BackwardContext& bctx,
          const OpNode& node,
          const std::vector<void*>& grad_outs,
          std::size_t arity,
          Ctx& c) {
    if (node.inputs.size() < arity || grad_outs.empty() || grad_outs[0] == nullptr)
        return false;
    c.g = (__bridge MPSGraph*)bctx.graph();
    c.go = as_tensor(grad_outs[0]);
    if (c.g == nil || c.go == nil)
        return false;
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
        if (node.inputs[i] < 0) {
            c.in.push_back(nil);
            continue;
        }
        MPSGraphTensor* t = as_tensor(bctx.forward(node.inputs[i]));
        if (t == nil)
            return false;
        c.in.push_back(cast_if_needed(c.g, t, c.go.dataType));
    }
    return true;
}

void give(BackwardContext& bctx, const OpNode& node, std::size_t i, MPSGraphTensor* grad) {
    if (i < node.inputs.size() && node.inputs[i] >= 0 && grad != nil)
        bctx.accumulate_grad(node.inputs[i], from_tensor(grad));
}

class DotVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "dot"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        Ctx c;
        if (!open(bctx, node, grad_outs, 2, c) || c.in[0].shape.count != 1 ||
            c.in[1].shape.count != 1)
            return false;
        give(bctx, node, 0,
             [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.in[1] name:nil]);
        give(bctx, node, 1,
             [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.in[0] name:nil]);
        return true;
    }
};

// inner (2-D) and outer (1-D) share y = a bᵀ once the 1-D operands of
// outer are read as columns.
template <bool OUTER>
class ABtVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return OUTER ? "outer" : "inner"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        Ctx c;
        if (!open(bctx, node, grad_outs, 2, c))
            return false;
        MPSGraphTensor* a = c.in[0];
        MPSGraphTensor* b = c.in[1];
        const NSUInteger want = OUTER ? 1 : 2;
        if (a.shape.count != want || b.shape.count != want || c.go.shape.count != 2 || !known(a) ||
            !known(b))
            return false;
        if (OUTER) {
            a = [c.g reshapeTensor:a withShape:@[ a.shape[0], @1 ] name:nil];
            b = [c.g reshapeTensor:b withShape:@[ b.shape[0], @1 ] name:nil];
        }
        MPSGraphTensor* da = mm(c.g, c.go, b);
        MPSGraphTensor* db = mm(c.g, tr(c.g, c.go), a);
        if (OUTER) {
            da = [c.g reshapeTensor:da withShape:@[ da.shape[0] ] name:nil];
            db = [c.g reshapeTensor:db withShape:@[ db.shape[0] ] name:nil];
        }
        give(bctx, node, 0, da);
        give(bctx, node, 1, db);
        return true;
    }
};

class TensordotVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "tensordot"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        Ctx c;
        if (!open(bctx, node, grad_outs, 2, c) || !known(c.in[0]) || !known(c.in[1]))
            return false;
        auto axes = [&](const char* key, std::size_t rank, std::vector<NSInteger>& out) {
            auto it = node.attrs.find(key);
            if (it == node.attrs.end())
                return false;
            const auto* v = std::get_if<std::vector<std::int64_t>>(&it->second);
            if (v == nullptr)
                return false;
            for (std::int64_t d : *v) {
                const std::int64_t a = d < 0 ? d + (std::int64_t)rank : d;
                if (a < 0 || a >= (std::int64_t)rank)
                    return false;
                out.push_back((NSInteger)a);
            }
            return true;
        };
        MPSGraph* g = c.g;
        MPSGraphTensor* a = c.in[0];
        MPSGraphTensor* b = c.in[1];
        std::vector<NSInteger> ca, cb;
        if (!axes("axes_a", a.shape.count, ca) || !axes("axes_b", b.shape.count, cb) ||
            ca.size() != cb.size())
            return false;
        auto free_of = [](NSUInteger rank, const std::vector<NSInteger>& con) {
            std::vector<NSInteger> f;
            for (NSInteger i = 0; i < (NSInteger)rank; ++i)
                if (std::find(con.begin(), con.end(), i) == con.end())
                    f.push_back(i);
            return f;
        };
        const auto fa = free_of(a.shape.count, ca);
        const auto fb = free_of(b.shape.count, cb);
        auto prod = [](NSArray<NSNumber*>* s, const std::vector<NSInteger>& ax) {
            long long p = 1;
            for (NSInteger i : ax)
                p *= s[(NSUInteger)i].longLongValue;
            return p;
        };
        auto perm = [](const std::vector<NSInteger>& x, const std::vector<NSInteger>& y) {
            NSMutableArray<NSNumber*>* p = [NSMutableArray array];
            for (NSInteger i : x)
                [p addObject:@(i)];
            for (NSInteger i : y)
                [p addObject:@(i)];
            return p;
        };
        const long long m = prod(a.shape, fa), k = prod(a.shape, ca), n = prod(b.shape, fb);
        // A2 = a[free_a, con_a] (m, k); B2 = b[con_b, free_b] (k, n); G2 (m, n).
        NSArray<NSNumber*>* pa = perm(fa, ca);
        NSArray<NSNumber*>* pb = perm(cb, fb);
        MPSGraphTensor* a2 = [g reshapeTensor:[g transposeTensor:a permutation:pa name:nil]
                                    withShape:@[ @(m), @(k) ]
                                         name:nil];
        MPSGraphTensor* b2 = [g reshapeTensor:[g transposeTensor:b permutation:pb name:nil]
                                    withShape:@[ @(k), @(n) ]
                                         name:nil];
        MPSGraphTensor* g2 = [g reshapeTensor:c.go withShape:@[ @(m), @(n) ] name:nil];
        MPSGraphTensor* da2 = mm(g, g2, tr(g, b2));  // (m, k)
        MPSGraphTensor* db2 = mm(g, tr(g, a2), g2);  // (k, n)
        auto undo = [&](MPSGraphTensor* t, MPSGraphTensor* like, NSArray<NSNumber*>* p) {
            NSMutableArray<NSNumber*>* permuted_shape = [NSMutableArray array];
            for (NSNumber* i in p)
                [permuted_shape addObject:like.shape[i.unsignedIntegerValue]];
            MPSGraphTensor* r = [g reshapeTensor:t withShape:permuted_shape name:nil];
            NSMutableArray<NSNumber*>* inverse = [NSMutableArray arrayWithCapacity:p.count];
            for (NSUInteger i = 0; i < p.count; ++i)
                [inverse addObject:@0];
            for (NSUInteger i = 0; i < p.count; ++i)
                inverse[p[i].unsignedIntegerValue] = @(i);
            return [g transposeTensor:r permutation:inverse name:nil];
        };
        give(bctx, node, 0, undo(da2, a, pa));
        give(bctx, node, 1, undo(db2, b, pb));
        return true;
    }
};

// y[n, o] = Σ_ij x1[n, i] W[o, i, j] x2[n, j] (+ b[o]).
class BilinearVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "bilinear_layer"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        Ctx c;
        if (!open(bctx, node, grad_outs, 3, c))
            return false;
        MPSGraph* g = c.g;
        MPSGraphTensor* x1 = c.in[0];
        MPSGraphTensor* x2 = c.in[1];
        MPSGraphTensor* w = c.in[2];
        if (x1 == nil || x2 == nil || w == nil || x1.shape.count != 2 || x2.shape.count != 2 ||
            w.shape.count != 3 || c.go.shape.count != 2 || !known(x1) || !known(x2) || !known(w))
            return false;
        NSNumber* N = x1.shape[0];
        NSNumber* I = x1.shape[1];
        NSNumber* J = x2.shape[1];
        NSNumber* O = w.shape[0];
        const long long oi = O.longLongValue * I.longLongValue;
        const long long oj = O.longLongValue * J.longLongValue;
        const long long ij = I.longLongValue * J.longLongValue;
        MPSGraphTensor* g3 = [g reshapeTensor:c.go withShape:@[ N, @1, O ] name:nil];
        // dx1[n, i] = Σ_o g[n, o] Σ_j W[o, i, j] x2[n, j]
        MPSGraphTensor* u = mm(g, x2, tr(g, [g reshapeTensor:w withShape:@[ @(oi), J ] name:nil]));
        u = [g reshapeTensor:u withShape:@[ N, O, I ] name:nil];
        MPSGraphTensor* dx1 = [g reshapeTensor:mm(g, g3, u) withShape:@[ N, I ] name:nil];
        // dx2[n, j] = Σ_o g[n, o] Σ_i x1[n, i] W[o, i, j]
        MPSGraphTensor* w_ioj = [g transposeTensor:w
                                         dimension:0
                                     withDimension:1
                                              name:nil];  // (I, O, J)
        MPSGraphTensor* v = mm(g, x1, [g reshapeTensor:w_ioj withShape:@[ I, @(oj) ] name:nil]);
        v = [g reshapeTensor:v withShape:@[ N, O, J ] name:nil];
        MPSGraphTensor* dx2 = [g reshapeTensor:mm(g, g3, v) withShape:@[ N, J ] name:nil];
        // dW[o, i, j] = Σ_n g[n, o] x1[n, i] x2[n, j]
        MPSGraphTensor* x1x2 =
            [g multiplicationWithPrimaryTensor:[g reshapeTensor:x1 withShape:@[ N, I, @1 ] name:nil]
                               secondaryTensor:[g reshapeTensor:x2 withShape:@[ N, @1, J ] name:nil]
                                          name:nil];
        MPSGraphTensor* dw =
            mm(g, tr(g, c.go), [g reshapeTensor:x1x2 withShape:@[ N, @(ij) ] name:nil]);
        dw = [g reshapeTensor:dw withShape:@[ O, I, J ] name:nil];
        give(bctx, node, 0, dx1);
        give(bctx, node, 1, dx2);
        give(bctx, node, 2, dw);
        if (node.inputs.size() > 3)
            give(bctx, node, 3, [g reductionSumWithTensor:c.go axis:0 name:nil]);
        return true;
    }
};

class TraceVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "trace"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 2 || !known(x))
                    return nil;
                NSArray<NSNumber*>* shape = x.shape;
                MPSGraphTensor* r = [g coordinateAlongAxis:0 withShape:shape name:nil];
                MPSGraphTensor* col = [g coordinateAlongAxis:1 withShape:shape name:nil];
                MPSGraphTensor* eye = [g castTensor:[g equalWithPrimaryTensor:r
                                                              secondaryTensor:col
                                                                         name:nil]
                                             toType:go.dataType
                                               name:nil];
                return [g multiplicationWithPrimaryTensor:eye
                                          secondaryTensor:[g reshapeTensor:go
                                                                 withShape:@[]
                                                                      name:nil]
                                                     name:@"trace_vjp"];
            });
    }
};

// 2-D only: the k-th diagonal of (rows, cols) — offset k > 0 above the main.
class DiagonalVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "diagonal"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        auto get = [&](const char* key, std::int64_t def) {
            auto it = node.attrs.find(key);
            if (it == node.attrs.end())
                return def;
            const auto* v = std::get_if<std::int64_t>(&it->second);
            return v ? *v : def;
        };
        const std::int64_t offset = get("offset", 0);
        const std::int64_t a1 = get("axis1", 0);
        const std::int64_t a2 = get("axis2", 1);
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 2 || go.shape.count != 1 || !known(x) ||
                    !((a1 == 0 && a2 == 1) || (a1 == 1 && a2 == 0)))
                    return nil;
                // With the axes swapped the diagonal of xᵀ is taken.
                const bool swapped = a1 == 1;
                NSArray<NSNumber*>* shape = x.shape;
                const long long rows = shape[swapped ? 1 : 0].longLongValue;
                const long long cols = shape[swapped ? 0 : 1].longLongValue;
                const long long len = go.shape[0].longLongValue;
                NSArray<NSNumber*>* rc = @[ @(rows), @(cols) ];
                MPSGraphTensor* r = [g coordinateAlongAxis:0 withShape:rc name:nil];
                MPSGraphTensor* col = [g coordinateAlongAxis:1 withShape:rc name:nil];
                MPSGraphTensor* on =
                    [g equalWithPrimaryTensor:
                            [g additionWithPrimaryTensor:r
                                         secondaryTensor:[g constantWithScalar:(double)offset
                                                                      dataType:r.dataType]
                                                    name:nil]
                              secondaryTensor:col
                                         name:nil];
                MPSGraphTensor* mask = [g castTensor:on toType:go.dataType name:nil];
                // g[t] sits at row t (offset ≥ 0) or column t (offset < 0).
                const bool by_row = offset >= 0;
                const long long span = by_row ? rows : cols;
                MPSGraphTensor* gp =
                    pad_or_identity(g, go, @[ @0 ], @[ @(span > len ? span - len : 0) ],
                                    MPSGraphPaddingModeConstant, 0.0);
                gp = [g reshapeTensor:gp
                            withShape:by_row ? @[ @(rows), @1 ] : @[ @1, @(cols) ]
                                 name:nil];
                MPSGraphTensor* dx = [g multiplicationWithPrimaryTensor:mask
                                                        secondaryTensor:gp
                                                                   name:nil];
                return swapped ? tr(g, dx) : dx;
            });
    }
};

class InvVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "inv"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (node.outputs.empty())
                    return nil;
                MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
                if (y == nil)
                    return nil;
                MPSGraphTensor* yt = tr(g, cast_if_needed(g, y, go.dataType));
                return [g negativeWithTensor:mm(g, mm(g, yt, go), yt) name:@"inv_vjp"];
            });
    }
};

// cof(A) = det(A) · A⁻ᵀ, written out for the sizes the forward lowers.
class DetVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "det"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(
            bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 2 || !known(x))
                    return nil;
                const long long n = x.shape[0].longLongValue;
                if (x.shape[1].longLongValue != n || (n != 1 && n != 2 && n != 3))
                    return nil;
                if (n == 1)
                    return [g reshapeTensor:go withShape:@[ @1, @1 ] name:nil];
                auto el = [&](NSInteger r, NSInteger c) {
                    MPSGraphTensor* row = [g sliceTensor:x dimension:0 start:r length:1 name:nil];
                    return [g sliceTensor:row dimension:1 start:c length:1 name:nil];  // (1, 1)
                };
                auto mul = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
                    return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
                };
                auto sub = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
                    return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
                };
                NSMutableArray<MPSGraphTensor*>* rows = [NSMutableArray array];
                if (n == 2) {
                    MPSGraphTensor* a = el(0, 0);
                    MPSGraphTensor* b = el(0, 1);
                    MPSGraphTensor* c = el(1, 0);
                    MPSGraphTensor* d = el(1, 1);
                    [rows addObject:[g concatTensors:@[ d, [g negativeWithTensor:c name:nil] ]
                                           dimension:1
                                                name:nil]];
                    [rows addObject:[g concatTensors:@[ [g negativeWithTensor:b name:nil], a ]
                                           dimension:1
                                                name:nil]];
                } else {
                    for (NSInteger i = 0; i < 3; ++i) {
                        NSMutableArray<MPSGraphTensor*>* row = [NSMutableArray array];
                        for (NSInteger j = 0; j < 3; ++j) {
                            const NSInteger r0 = (i + 1) % 3, r1 = (i + 2) % 3;
                            const NSInteger c0 = (j + 1) % 3, c1 = (j + 2) % 3;
                            // Cyclic minors carry the cofactor sign already.
                            [row addObject:sub(mul(el(r0, c0), el(r1, c1)),
                                               mul(el(r0, c1), el(r1, c0)))];
                        }
                        [rows addObject:[g concatTensors:row dimension:1 name:nil]];
                    }
                }
                MPSGraphTensor* cof = [g concatTensors:rows dimension:0 name:nil];
                return mul(cof, [g reshapeTensor:go withShape:@[] name:nil]);
            });
    }
};

class SolveTriangularVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "solve_triangular"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        if (node.outputs.empty())
            return false;
        MPSGraphTensor* x = as_tensor(bctx.forward(node.outputs[0].id));
        Ctx c;
        if (!open(bctx, node, grad_outs, 2, c) || x == nil)
            return false;
        const bool upper = bool_attr(node, "upper", false);
        const bool unit = bool_attr(node, "unitriangular", false);
        MPSGraphTensor* db = emit_solve_triangular_mps(c.g, tr(c.g, c.in[0]), c.go, !upper, unit);
        if (db == nil)
            return false;
        MPSGraphTensor* x_cast = cast_if_needed(c.g, x, c.go.dataType);
        MPSGraphTensor* da = [c.g negativeWithTensor:mm(c.g, db, tr(c.g, x_cast)) name:nil];
        da = [c.g bandPartWithTensor:da numLower:(upper ? 0 : -1)numUpper:(upper ? -1 : 0)name:nil];
        if (unit) {
            MPSGraphTensor* diag = [c.g bandPartWithTensor:da numLower:0 numUpper:0 name:nil];
            da = [c.g subtractionWithPrimaryTensor:da secondaryTensor:diag name:nil];
        }
        give(bctx, node, 0, da);
        give(bctx, node, 1, db);
        return true;
    }
};

struct ProductsVjpRegistrar {
    ProductsVjpRegistrar() {
        register_vjp_emitter(std::make_unique<DotVjp>());
        register_vjp_emitter(std::make_unique<ABtVjp<false>>());
        register_vjp_emitter(std::make_unique<ABtVjp<true>>());
        register_vjp_emitter(std::make_unique<TensordotVjp>());
        register_vjp_emitter(std::make_unique<BilinearVjp>());
        register_vjp_emitter(std::make_unique<TraceVjp>());
        register_vjp_emitter(std::make_unique<DiagonalVjp>());
        register_vjp_emitter(std::make_unique<InvVjp>());
        register_vjp_emitter(std::make_unique<DetVjp>());
        register_vjp_emitter(std::make_unique<SolveTriangularVjp>());
    }
};

[[maybe_unused]] static const ProductsVjpRegistrar g_products_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
