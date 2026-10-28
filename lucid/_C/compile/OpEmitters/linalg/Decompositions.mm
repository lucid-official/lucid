// lucid/_C/compile/OpEmitters/linalg/Decompositions.mm
//
// MPSGraph emitters for linear-algebra decompositions and triangular solves:
//
//   - ``solve_triangular`` — unrolled row-by-row forward / back substitution
//   - ``cholesky``         — unrolled column-by-column Cholesky-Banachiewicz
//   - ``qr``               — unrolled Householder QR (LAPACK slarfg reflector sign)
//   - ``eigh``             — cyclic Jacobi symmetric eigendecomposition
//   - ``svd``              — one-sided Hestenes-Jacobi singular value decomposition
//   - ``pinv``             — Moore-Penrose pseudoinverse via one-sided Jacobi SVD

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "../_AttrHelpers.h"
#include "../_SortOrder.h"
#include "Decompositions.h"

namespace lucid::compile {

namespace {

constexpr std::int64_t kMaxDirectDim = 32;
constexpr std::int64_t kMaxJacobiDim = 16;

MPSGraphTensor* to_f32(MPSGraph* g, MPSGraphTensor* t) {
    return t.dataType == MPSDataTypeFloat32 ? t
                                            : [g castTensor:t toType:MPSDataTypeFloat32 name:nil];
}

MPSGraphTensor* from_f32(MPSGraph* g, MPSGraphTensor* t, MPSDataType dt) {
    return t.dataType == dt ? t : [g castTensor:t toType:dt name:nil];
}

bool valid_matrix_tensor(MPSGraphTensor* x, std::int64_t max_dim) {
    if (x == nil || !(x.dataType & MPSDataTypeFloatBit) || x.shape.count < 2)
        return false;
    for (NSNumber* d in x.shape) {
        if (d.longLongValue <= 0)
            return false;
    }
    const NSUInteger r = x.shape.count;
    return x.shape[r - 2].longLongValue <= max_dim && x.shape[r - 1].longLongValue <= max_dim;
}

MPSShape* make_batch_shape(MPSGraphTensor* like, std::int64_t rows, std::int64_t cols) {
    NSMutableArray<NSNumber*>* sh = [NSMutableArray arrayWithCapacity:like.shape.count];
    for (NSUInteger d = 0; d + 2 < like.shape.count; ++d)
        [sh addObject:like.shape[d]];
    [sh addObject:@(rows)];
    [sh addObject:@(cols)];
    return sh;
}

MPSGraphTensor*
batch_const(MPSGraph* g, MPSGraphTensor* like, double val, std::int64_t rows, std::int64_t cols) {
    return [g constantWithScalar:val
                           shape:make_batch_shape(like, rows, cols)
                        dataType:like.dataType];
}

MPSGraphTensor* swap_last_two(MPSGraph* g, MPSGraphTensor* t) {
    const NSInteger r = (NSInteger)t.shape.count;
    return [g transposeTensor:t dimension:r - 2 withDimension:r - 1 name:nil];
}

MPSGraphTensor*
concat_along(MPSGraph* g, const std::vector<MPSGraphTensor*>& parts, NSInteger dim) {
    if (parts.size() == 1)
        return parts[0];
    NSMutableArray<MPSGraphTensor*>* arr = [NSMutableArray arrayWithCapacity:parts.size()];
    for (MPSGraphTensor* p : parts)
        [arr addObject:p];
    return [g concatTensors:arr dimension:dim name:nil];
}

std::vector<MPSGraphTensor*>
make_identity_cols(MPSGraph* g, MPSGraphTensor* like, std::int64_t m, std::int64_t k) {
    const NSInteger r_dim = (NSInteger)like.shape.count - 2;
    std::vector<MPSGraphTensor*> cols;
    cols.reserve((std::size_t)k);
    for (std::int64_t j = 0; j < k; ++j) {
        std::vector<MPSGraphTensor*> pieces;
        if (j > 0)
            pieces.push_back(batch_const(g, like, 0.0, j, 1));
        pieces.push_back(batch_const(g, like, 1.0, 1, 1));
        if (j + 1 < m)
            pieces.push_back(batch_const(g, like, 0.0, m - j - 1, 1));
        cols.push_back(concat_along(g, pieces, r_dim));
    }
    return cols;
}

// ── solve_triangular ─────────────────────────────────────────────────

MPSGraphTensor* solve_triangular_row(MPSGraph* g,
                                     MPSGraphTensor* a,
                                     MPSGraphTensor* b,
                                     const std::vector<MPSGraphTensor*>& x_rows,
                                     std::int64_t i,
                                     std::int64_t n,
                                     bool upper,
                                     bool unit) {
    const NSInteger r_dim = (NSInteger)a.shape.count - 2;
    const NSInteger c_dim = (NSInteger)a.shape.count - 1;
    MPSGraphTensor* row_a = [g sliceTensor:a dimension:r_dim start:i length:1 name:nil];
    MPSGraphTensor* rhs = [g sliceTensor:b dimension:r_dim start:i length:1 name:nil];
    const std::int64_t j_begin = upper ? i + 1 : 0;
    const std::int64_t j_end = upper ? n : i;
    for (std::int64_t j = j_begin; j < j_end; ++j) {
        MPSGraphTensor* a_ij = [g sliceTensor:row_a dimension:c_dim start:j length:1 name:nil];
        MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:a_ij
                                                  secondaryTensor:x_rows[j]
                                                             name:nil];
        rhs = [g subtractionWithPrimaryTensor:rhs secondaryTensor:prod name:nil];
    }
    if (!unit) {
        MPSGraphTensor* a_ii = [g sliceTensor:row_a dimension:c_dim start:i length:1 name:nil];
        rhs = [g divisionWithPrimaryTensor:rhs secondaryTensor:a_ii name:nil];
    }
    return rhs;
}

// ── cholesky ─────────────────────────────────────────────────────────

MPSGraphTensor* cholesky_col(MPSGraph* g,
                             MPSGraphTensor* a,
                             const std::vector<MPSGraphTensor*>& cols,
                             std::int64_t j,
                             std::int64_t n) {
    const NSInteger r_dim = (NSInteger)a.shape.count - 2;
    const NSInteger c_dim = (NSInteger)a.shape.count - 1;
    MPSGraphTensor* col_j = [g sliceTensor:a dimension:c_dim start:j length:1 name:nil];
    MPSGraphTensor* sub = [g sliceTensor:col_j dimension:r_dim start:j length:n - j name:nil];
    for (std::int64_t k = 0; k < j; ++k) {
        MPSGraphTensor* l_jk = [g sliceTensor:cols[k] dimension:r_dim start:j length:1 name:nil];
        MPSGraphTensor* l_ik = [g sliceTensor:cols[k]
                                    dimension:r_dim
                                        start:j
                                       length:n - j
                                         name:nil];
        MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:l_ik
                                                  secondaryTensor:l_jk
                                                             name:nil];
        sub = [g subtractionWithPrimaryTensor:sub secondaryTensor:prod name:nil];
    }
    MPSGraphTensor* diag_sq = [g sliceTensor:sub dimension:r_dim start:0 length:1 name:nil];
    MPSGraphTensor* l_jj = [g squareRootWithTensor:diag_sq name:nil];
    std::vector<MPSGraphTensor*> pieces;
    if (j > 0)
        pieces.push_back(batch_const(g, a, 0.0, j, 1));
    pieces.push_back(l_jj);
    if (j + 1 < n) {
        MPSGraphTensor* tail = [g sliceTensor:sub
                                    dimension:r_dim
                                        start:1
                                       length:n - j - 1
                                         name:nil];
        pieces.push_back([g divisionWithPrimaryTensor:tail secondaryTensor:l_jj name:nil]);
    }
    return concat_along(g, pieces, r_dim);
}

MPSGraphTensor* emit_cholesky_lower(MPSGraph* g, MPSGraphTensor* a, std::int64_t n) {
    const NSInteger c_dim = (NSInteger)a.shape.count - 1;
    std::vector<MPSGraphTensor*> cols;
    cols.reserve((std::size_t)n);
    for (std::int64_t j = 0; j < n; ++j)
        cols.push_back(cholesky_col(g, a, cols, j, n));
    return concat_along(g, cols, c_dim);
}

// ── qr (Householder with LAPACK slarfg reflector sign) ───────────────

std::pair<MPSGraphTensor*, MPSGraphTensor*>
householder_reflector(MPSGraph* g, MPSGraphTensor* x, std::int64_t mk) {
    const NSInteger r_dim = (NSInteger)x.shape.count - 2;
    MPSGraphTensor* x0 = [g sliceTensor:x dimension:r_dim start:0 length:1 name:nil];
    MPSGraphTensor* x_tail = [g sliceTensor:x dimension:r_dim start:1 length:mk - 1 name:nil];
    MPSGraphTensor* tail_sq = [g reductionSumWithTensor:[g squareWithTensor:x_tail name:nil]
                                                   axes:@[ @(r_dim) ]
                                                   name:nil];
    MPSGraphTensor* norm_x =
        [g squareRootWithTensor:[g additionWithPrimaryTensor:[g squareWithTensor:x0 name:nil]
                                             secondaryTensor:tail_sq
                                                        name:nil]
                           name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* neg_one = [g constantWithScalar:-1.0 dataType:x.dataType];
    MPSGraphTensor* sign_x0 =
        [g selectWithPredicateTensor:[g greaterThanOrEqualToWithPrimaryTensor:x0
                                                              secondaryTensor:zero
                                                                         name:nil]
                 truePredicateTensor:one
                falsePredicateTensor:neg_one
                                name:nil];
    MPSGraphTensor* beta = [g negativeWithTensor:[g multiplicationWithPrimaryTensor:sign_x0
                                                                    secondaryTensor:norm_x
                                                                               name:nil]
                                            name:nil];
    MPSGraphTensor* has_tail = [g greaterThanWithPrimaryTensor:tail_sq
                                               secondaryTensor:zero
                                                          name:nil];
    MPSGraphTensor* x0_minus_beta = [g subtractionWithPrimaryTensor:x0
                                                    secondaryTensor:beta
                                                               name:nil];
    MPSGraphTensor* safe_denom = [g selectWithPredicateTensor:has_tail
                                          truePredicateTensor:x0_minus_beta
                                         falsePredicateTensor:one
                                                         name:nil];
    MPSGraphTensor* safe_beta = [g selectWithPredicateTensor:has_tail
                                         truePredicateTensor:beta
                                        falsePredicateTensor:one
                                                        name:nil];
    MPSGraphTensor* tau_raw = [g divisionWithPrimaryTensor:[g subtractionWithPrimaryTensor:beta
                                                                           secondaryTensor:x0
                                                                                      name:nil]
                                           secondaryTensor:safe_beta
                                                      name:nil];
    MPSGraphTensor* tau = [g selectWithPredicateTensor:has_tail
                                   truePredicateTensor:tau_raw
                                  falsePredicateTensor:zero
                                                  name:nil];
    MPSGraphTensor* v_tail = [g divisionWithPrimaryTensor:x_tail
                                          secondaryTensor:safe_denom
                                                     name:nil];
    MPSGraphTensor* v = concat_along(g, {batch_const(g, x, 1.0, 1, 1), v_tail}, r_dim);
    return {v, tau};
}

MPSGraphTensor* apply_reflector_left(MPSGraph* g,
                                     MPSGraphTensor* mat,
                                     MPSGraphTensor* v,
                                     MPSGraphTensor* tau,
                                     std::int64_t k,
                                     std::int64_t m) {
    const NSInteger r_dim = (NSInteger)mat.shape.count - 2;
    MPSGraphTensor* bot = [g sliceTensor:mat dimension:r_dim start:k length:m - k name:nil];
    MPSGraphTensor* vt_bot = [g matrixMultiplicationWithPrimaryTensor:swap_last_two(g, v)
                                                      secondaryTensor:bot
                                                                 name:nil];
    MPSGraphTensor* tau_v = [g multiplicationWithPrimaryTensor:tau secondaryTensor:v name:nil];
    MPSGraphTensor* upd = [g matrixMultiplicationWithPrimaryTensor:tau_v
                                                   secondaryTensor:vt_bot
                                                              name:nil];
    MPSGraphTensor* bot_new = [g subtractionWithPrimaryTensor:bot secondaryTensor:upd name:nil];
    if (k == 0)
        return bot_new;
    MPSGraphTensor* top = [g sliceTensor:mat dimension:r_dim start:0 length:k name:nil];
    return concat_along(g, {top, bot_new}, r_dim);
}

std::pair<MPSGraphTensor*, MPSGraphTensor*>
emit_qr_factors(MPSGraph* g, MPSGraphTensor* a, std::int64_t m, std::int64_t n) {
    const NSInteger r_dim = (NSInteger)a.shape.count - 2;
    const NSInteger c_dim = (NSInteger)a.shape.count - 1;
    const std::int64_t k_min = std::min(m, n);
    MPSGraphTensor* r_mat = a;
    std::vector<MPSGraphTensor*> vs(k_min, nil);
    std::vector<MPSGraphTensor*> taus(k_min, nil);
    for (std::int64_t k = 0; k < k_min; ++k) {
        const std::int64_t mk = m - k;
        if (mk <= 1)
            continue;
        MPSGraphTensor* col_k = [g sliceTensor:r_mat dimension:c_dim start:k length:1 name:nil];
        MPSGraphTensor* x = [g sliceTensor:col_k dimension:r_dim start:k length:mk name:nil];
        auto [v, tau] = householder_reflector(g, x, mk);
        vs[k] = v;
        taus[k] = tau;
        r_mat = apply_reflector_left(g, r_mat, v, tau, k, m);
    }
    MPSGraphTensor* q_mat = concat_along(g, make_identity_cols(g, a, m, k_min), c_dim);
    for (std::int64_t k = k_min - 1; k >= 0; --k) {
        if (vs[k] != nil)
            q_mat = apply_reflector_left(g, q_mat, vs[k], taus[k], k, m);
    }
    MPSGraphTensor* r_sliced =
        m > k_min ? [g sliceTensor:r_mat dimension:r_dim start:0 length:k_min name:nil] : r_mat;
    MPSGraphTensor* r_upper =
        k_min > 1 ? [g bandPartWithTensor:r_sliced numLower:0 numUpper:-1 name:nil] : r_sliced;
    return {q_mat, r_upper};
}

// ── Jacobi helpers (eigh, svd, pinv) ─────────────────────────────────

int jacobi_sweeps(std::int64_t n) {
    if (n <= 2)
        return 1;
    return n <= 4 ? 7 : (n <= 8 ? 8 : 9);
}

std::pair<MPSGraphTensor*, MPSGraphTensor*>
jacobi_cs(MPSGraph* g, MPSGraphTensor* app, MPSGraphTensor* aqq, MPSGraphTensor* apq) {
    MPSGraphTensor* eps = [g constantWithScalar:1e-12 dataType:apq.dataType];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:apq.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:apq.dataType];
    MPSGraphTensor* neg_one = [g constantWithScalar:-1.0 dataType:apq.dataType];
    MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:apq.dataType];
    MPSGraphTensor* active = [g greaterThanWithPrimaryTensor:[g absoluteWithTensor:apq name:nil]
                                             secondaryTensor:eps
                                                        name:nil];
    MPSGraphTensor* two_apq = [g multiplicationWithPrimaryTensor:two secondaryTensor:apq name:nil];
    MPSGraphTensor* safe_denom = [g selectWithPredicateTensor:active
                                          truePredicateTensor:two_apq
                                         falsePredicateTensor:one
                                                         name:nil];
    MPSGraphTensor* tau = [g divisionWithPrimaryTensor:[g subtractionWithPrimaryTensor:aqq
                                                                       secondaryTensor:app
                                                                                  name:nil]
                                       secondaryTensor:safe_denom
                                                  name:nil];
    MPSGraphTensor* sign_tau =
        [g selectWithPredicateTensor:[g greaterThanOrEqualToWithPrimaryTensor:tau
                                                              secondaryTensor:zero
                                                                         name:nil]
                 truePredicateTensor:one
                falsePredicateTensor:neg_one
                                name:nil];
    MPSGraphTensor* hyp =
        [g squareRootWithTensor:[g additionWithPrimaryTensor:one
                                             secondaryTensor:[g squareWithTensor:tau name:nil]
                                                        name:nil]
                           name:nil];
    MPSGraphTensor* t_raw = [g
        divisionWithPrimaryTensor:sign_tau
                  secondaryTensor:[g additionWithPrimaryTensor:[g absoluteWithTensor:tau name:nil]
                                               secondaryTensor:hyp
                                                          name:nil]
                             name:nil];
    MPSGraphTensor* t = [g selectWithPredicateTensor:active
                                 truePredicateTensor:t_raw
                                falsePredicateTensor:zero
                                                name:nil];
    MPSGraphTensor* c = [g
        divisionWithPrimaryTensor:one
                  secondaryTensor:
                      [g squareRootWithTensor:[g additionWithPrimaryTensor:one
                                                           secondaryTensor:[g squareWithTensor:t
                                                                                          name:nil]
                                                                      name:nil]
                                         name:nil]
                             name:nil];
    MPSGraphTensor* s = [g multiplicationWithPrimaryTensor:t secondaryTensor:c name:nil];
    return {c, s};
}

std::pair<MPSGraphTensor*, MPSGraphTensor*> rotate_pair(
    MPSGraph* g, MPSGraphTensor* xp, MPSGraphTensor* xq, MPSGraphTensor* c, MPSGraphTensor* s) {
    MPSGraphTensor* np = [g subtractionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:c
                                                                            secondaryTensor:xp
                                                                                       name:nil]
                                         secondaryTensor:[g multiplicationWithPrimaryTensor:s
                                                                            secondaryTensor:xq
                                                                                       name:nil]
                                                    name:nil];
    MPSGraphTensor* nq = [g
        additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:s secondaryTensor:xp name:nil]
                  secondaryTensor:[g multiplicationWithPrimaryTensor:c secondaryTensor:xq name:nil]
                             name:nil];
    return {np, nq};
}

MPSGraphTensor* replace_two_slices(MPSGraph* g,
                                   MPSGraphTensor* t,
                                   NSInteger dim,
                                   std::int64_t n,
                                   std::int64_t p,
                                   std::int64_t q,
                                   MPSGraphTensor* sp,
                                   MPSGraphTensor* sq) {
    std::vector<MPSGraphTensor*> parts;
    if (p > 0)
        parts.push_back([g sliceTensor:t dimension:dim start:0 length:p name:nil]);
    parts.push_back(sp);
    if (q > p + 1)
        parts.push_back([g sliceTensor:t dimension:dim start:p + 1 length:q - p - 1 name:nil]);
    parts.push_back(sq);
    if (n > q + 1)
        parts.push_back([g sliceTensor:t dimension:dim start:q + 1 length:n - q - 1 name:nil]);
    return concat_along(g, parts, dim);
}

std::pair<MPSGraphTensor*, MPSGraphTensor*>
sort_values_and_cols(MPSGraph* g, MPSGraphTensor* vals, MPSGraphTensor* mat, bool descending) {
    const NSInteger rank = (NSInteger)mat.shape.count;
    MPSGraphTensor* order = graph_sort_order(g, vals, rank - 2, descending);
    MPSGraphTensor* sorted_vals = graph_gather_sorted(g, vals, order, rank - 2);
    MPSGraphTensor* order_exp = [g expandDimsOfTensor:order axes:@[ @(rank - 2) ] name:nil];
    MPSGraphTensor* order_full = [g broadcastTensor:order_exp toShape:mat.shape name:nil];
    MPSGraphTensor* sorted_mat = graph_gather_sorted(g, mat, order_full, rank - 1);
    return {sorted_vals, sorted_mat};
}

// ── eigh ─────────────────────────────────────────────────────────────

MPSGraphTensor* symmetrize_from_triangle(MPSGraph* g, MPSGraphTensor* a, bool upper) {
    MPSGraphTensor* tri = [g bandPartWithTensor:a
                                       numLower:(upper ? 0 : -1)numUpper:(upper ? -1 : 0)name:nil];
    MPSGraphTensor* diag = [g bandPartWithTensor:a numLower:0 numUpper:0 name:nil];
    MPSGraphTensor* sum = [g additionWithPrimaryTensor:tri
                                       secondaryTensor:swap_last_two(g, tri)
                                                  name:nil];
    return [g subtractionWithPrimaryTensor:sum secondaryTensor:diag name:nil];
}

MPSGraphTensor* eigh_jacobi_step(MPSGraph* g,
                                 MPSGraphTensor* s_mat,
                                 std::vector<MPSGraphTensor*>& v_cols,
                                 std::int64_t n,
                                 std::int64_t p,
                                 std::int64_t q) {
    const NSInteger r_dim = (NSInteger)s_mat.shape.count - 2;
    const NSInteger c_dim = (NSInteger)s_mat.shape.count - 1;
    MPSGraphTensor* rp = [g sliceTensor:s_mat dimension:r_dim start:p length:1 name:nil];
    MPSGraphTensor* rq = [g sliceTensor:s_mat dimension:r_dim start:q length:1 name:nil];
    MPSGraphTensor* app = [g sliceTensor:rp dimension:c_dim start:p length:1 name:nil];
    MPSGraphTensor* apq = [g sliceTensor:rp dimension:c_dim start:q length:1 name:nil];
    MPSGraphTensor* aqq = [g sliceTensor:rq dimension:c_dim start:q length:1 name:nil];
    auto [c, s] = jacobi_cs(g, app, aqq, apq);
    std::tie(rp, rq) = rotate_pair(g, rp, rq, c, s);
    s_mat = replace_two_slices(g, s_mat, r_dim, n, p, q, rp, rq);
    MPSGraphTensor* cp = [g sliceTensor:s_mat dimension:c_dim start:p length:1 name:nil];
    MPSGraphTensor* cq = [g sliceTensor:s_mat dimension:c_dim start:q length:1 name:nil];
    std::tie(cp, cq) = rotate_pair(g, cp, cq, c, s);
    s_mat = replace_two_slices(g, s_mat, c_dim, n, p, q, cp, cq);
    std::tie(v_cols[p], v_cols[q]) = rotate_pair(g, v_cols[p], v_cols[q], c, s);
    return s_mat;
}

std::pair<MPSGraphTensor*, MPSGraphTensor*>
emit_eigh_factors(MPSGraph* g, MPSGraphTensor* a, std::int64_t n, bool upper) {
    const NSInteger rank = (NSInteger)a.shape.count;
    MPSGraphTensor* s_mat = symmetrize_from_triangle(g, a, upper);
    auto v_cols = make_identity_cols(g, a, n, n);
    const int sweeps = jacobi_sweeps(n);
    for (int sw = 0; sw < sweeps && n > 1; ++sw) {
        for (std::int64_t p = 0; p < n - 1; ++p) {
            for (std::int64_t q = p + 1; q < n; ++q)
                s_mat = eigh_jacobi_step(g, s_mat, v_cols, n, p, q);
        }
    }
    MPSGraphTensor* diag = [g bandPartWithTensor:s_mat numLower:0 numUpper:0 name:nil];
    MPSGraphTensor* w_keep = [g reductionSumWithTensor:diag axes:@[ @(rank - 1) ] name:nil];
    NSMutableArray<NSNumber*>* w_sh = [NSMutableArray arrayWithCapacity:(NSUInteger)(rank - 1)];
    for (NSInteger d = 0; d < rank - 2; ++d)
        [w_sh addObject:a.shape[(NSUInteger)d]];
    [w_sh addObject:@(n)];
    MPSGraphTensor* w = [g reshapeTensor:w_keep withShape:w_sh name:nil];
    MPSGraphTensor* v_mat = concat_along(g, v_cols, rank - 1);
    if (n <= 1)
        return {w, v_mat};
    return sort_values_and_cols(g, w, v_mat, false);
}

// ── svd & pinv (One-sided Hestenes-Jacobi) ───────────────────────────

void svd_one_sided_step(MPSGraph* g,
                        std::vector<MPSGraphTensor*>& w_cols,
                        std::vector<MPSGraphTensor*>& v_cols,
                        std::int64_t p,
                        std::int64_t q,
                        NSInteger r_dim) {
    MPSGraphTensor* wp = w_cols[p];
    MPSGraphTensor* wq = w_cols[q];
    MPSGraphTensor* app = [g reductionSumWithTensor:[g squareWithTensor:wp name:nil]
                                               axes:@[ @(r_dim) ]
                                               name:nil];
    MPSGraphTensor* aqq = [g reductionSumWithTensor:[g squareWithTensor:wq name:nil]
                                               axes:@[ @(r_dim) ]
                                               name:nil];
    MPSGraphTensor* apq = [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:wp
                                                                       secondaryTensor:wq
                                                                                  name:nil]
                                               axes:@[ @(r_dim) ]
                                               name:nil];
    auto [c, s] = jacobi_cs(g, app, aqq, apq);
    std::tie(w_cols[p], w_cols[q]) = rotate_pair(g, w_cols[p], w_cols[q], c, s);
    std::tie(v_cols[p], v_cols[q]) = rotate_pair(g, v_cols[p], v_cols[q], c, s);
}

std::tuple<MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*>
emit_svd_tall(MPSGraph* g, MPSGraphTensor* a, std::int64_t m, std::int64_t n) {
    const NSInteger rank = (NSInteger)a.shape.count;
    const NSInteger r_dim = rank - 2;
    const NSInteger c_dim = rank - 1;
    std::vector<MPSGraphTensor*> w_cols;
    w_cols.reserve((std::size_t)n);
    for (std::int64_t j = 0; j < n; ++j)
        w_cols.push_back([g sliceTensor:a dimension:c_dim start:j length:1 name:nil]);
    auto v_cols = make_identity_cols(g, a, n, n);
    const int sweeps = jacobi_sweeps(n);
    for (int sw = 0; sw < sweeps && n > 1; ++sw) {
        for (std::int64_t p = 0; p < n - 1; ++p) {
            for (std::int64_t q = p + 1; q < n; ++q)
                svd_one_sided_step(g, w_cols, v_cols, p, q, r_dim);
        }
    }
    MPSGraphTensor* w_mat = concat_along(g, w_cols, c_dim);
    MPSGraphTensor* v_mat = concat_along(g, v_cols, c_dim);
    MPSGraphTensor* s_keep =
        [g squareRootWithTensor:[g reductionSumWithTensor:[g squareWithTensor:w_mat name:nil]
                                                     axes:@[ @(r_dim) ]
                                                     name:nil]
                           name:nil];
    MPSGraphTensor* eps = [g constantWithScalar:1e-12 dataType:a.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:a.dataType];
    MPSGraphTensor* safe_s = [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:s_keep
                                                                          secondaryTensor:eps
                                                                                     name:nil]
                                      truePredicateTensor:s_keep
                                     falsePredicateTensor:one
                                                     name:nil];
    MPSGraphTensor* u_mat = [g divisionWithPrimaryTensor:w_mat secondaryTensor:safe_s name:nil];
    NSMutableArray<NSNumber*>* s_sh = [NSMutableArray arrayWithCapacity:(NSUInteger)(rank - 1)];
    for (NSInteger d = 0; d < rank - 2; ++d)
        [s_sh addObject:a.shape[(NSUInteger)d]];
    [s_sh addObject:@(n)];
    MPSGraphTensor* s_vec = [g reshapeTensor:s_keep withShape:s_sh name:nil];
    if (n > 1) {
        auto [s_sorted, u_sorted] = sort_values_and_cols(g, s_vec, u_mat, true);
        auto [_, v_sorted] = sort_values_and_cols(g, s_vec, v_mat, true);
        s_vec = s_sorted;
        u_mat = u_sorted;
        v_mat = v_sorted;
    }
    return {u_mat, s_vec, swap_last_two(g, v_mat)};
}

std::tuple<MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*>
emit_svd_any(MPSGraph* g, MPSGraphTensor* a, std::int64_t m, std::int64_t n) {
    if (m >= n)
        return emit_svd_tall(g, a, m, n);
    auto [u_t, s_vec, vh_t] = emit_svd_tall(g, swap_last_two(g, a), n, m);
    return {swap_last_two(g, vh_t), s_vec, swap_last_two(g, u_t)};
}

}  // namespace

MPSGraphTensor* emit_solve_triangular_mps(
    MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool upper, bool unitriangular) {
    if (!valid_matrix_tensor(a, kMaxDirectDim) || !valid_matrix_tensor(b, kMaxDirectDim))
        return nil;
    if (a.shape.count != b.shape.count)
        return nil;
    const NSUInteger rank = a.shape.count;
    const std::int64_t n = a.shape[rank - 2].longLongValue;
    if (a.shape[rank - 1].longLongValue != n || b.shape[rank - 2].longLongValue != n)
        return nil;
    const MPSDataType orig_dt = b.dataType;
    MPSGraphTensor* a_f = to_f32(g, a);
    MPSGraphTensor* b_f = to_f32(g, b);
    std::vector<MPSGraphTensor*> x_rows((std::size_t)n, nil);
    for (std::int64_t step = 0; step < n; ++step) {
        const std::int64_t i = upper ? (n - 1 - step) : step;
        x_rows[i] = solve_triangular_row(g, a_f, b_f, x_rows, i, n, upper, unitriangular);
    }
    MPSGraphTensor* out = concat_along(g, x_rows, (NSInteger)rank - 2);
    return from_f32(g, out, orig_dt);
}

namespace {

class SolveTriangularEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "solve_triangular"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty() || node.inputs[0] < 0 ||
            node.inputs[1] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        if (g == nil || a == nil || b == nil)
            return false;
        const bool upper = bool_attr(node, "upper", false);
        const bool unit = bool_attr(node, "unitriangular", false);
        MPSGraphTensor* out = emit_solve_triangular_mps(g, a, b, upper, unit);
        if (out == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
        return true;
    }
};

std::int64_t square_matrix_dim(MPSGraph* g, MPSGraphTensor* x, std::int64_t max_dim) {
    if (g == nil || !valid_matrix_tensor(x, max_dim))
        return 0;
    const NSUInteger rank = x.shape.count;
    const std::int64_t n = x.shape[rank - 2].longLongValue;
    return x.shape[rank - 1].longLongValue == n ? n : 0;
}

class CholeskyEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cholesky"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty() || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        const std::int64_t n = square_matrix_dim(g, x, kMaxDirectDim);
        if (n <= 0)
            return false;
        const bool upper = bool_attr(node, "upper", false);
        MPSGraphTensor* work = to_f32(g, x);
        if (upper)
            work = swap_last_two(g, work);
        MPSGraphTensor* l_mat = emit_cholesky_lower(g, work, n);
        if (upper)
            l_mat = swap_last_two(g, l_mat);
        ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, l_mat, x.dataType));
        return true;
    }
};

class QrEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "qr"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.size() < 2 || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        if (g == nil || !valid_matrix_tensor(x, kMaxDirectDim))
            return false;
        const NSUInteger rank = x.shape.count;
        const std::int64_t m = x.shape[rank - 2].longLongValue;
        const std::int64_t n = x.shape[rank - 1].longLongValue;
        const MPSDataType orig_dt = x.dataType;
        auto [q_mat, r_mat] = emit_qr_factors(g, to_f32(g, x), m, n);
        ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, q_mat, orig_dt));
        ctx.bind(node.outputs[1].id, (__bridge void*)from_f32(g, r_mat, orig_dt));
        return true;
    }
};

class EighEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "eigh"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.size() < 2 || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        const std::int64_t n = square_matrix_dim(g, x, kMaxJacobiDim);
        if (n <= 0)
            return false;
        auto [w, v_mat] = emit_eigh_factors(g, to_f32(g, x), n, bool_attr(node, "upper", false));
        ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, w, x.dataType));
        ctx.bind(node.outputs[1].id, (__bridge void*)from_f32(g, v_mat, x.dataType));
        return true;
    }
};

class SvdEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "svd"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty() || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        if (g == nil || !valid_matrix_tensor(x, kMaxJacobiDim))
            return false;
        const NSUInteger rank = x.shape.count;
        const std::int64_t m = x.shape[rank - 2].longLongValue;
        const std::int64_t n = x.shape[rank - 1].longLongValue;
        const bool compute_uv = bool_attr(node, "compute_uv", node.outputs.size() >= 3);
        const MPSDataType orig_dt = x.dataType;
        auto [u_mat, s_vec, vh_mat] = emit_svd_any(g, to_f32(g, x), m, n);
        if (!compute_uv || node.outputs.size() == 1) {
            ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, s_vec, orig_dt));
            return true;
        }
        if (node.outputs.size() < 3)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, u_mat, orig_dt));
        ctx.bind(node.outputs[1].id, (__bridge void*)from_f32(g, s_vec, orig_dt));
        ctx.bind(node.outputs[2].id, (__bridge void*)from_f32(g, vh_mat, orig_dt));
        return true;
    }
};

class PinvEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pinv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty() || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        if (g == nil || !valid_matrix_tensor(x, kMaxJacobiDim))
            return false;
        const NSInteger rank = (NSInteger)x.shape.count;
        const std::int64_t m = x.shape[(NSUInteger)(rank - 2)].longLongValue;
        const std::int64_t n = x.shape[(NSUInteger)(rank - 1)].longLongValue;
        const MPSDataType orig_dt = x.dataType;
        auto [u_mat, s_vec, vh_mat] = emit_svd_any(g, to_f32(g, x), m, n);
        MPSGraphTensor* max_s = [g reductionMaximumWithTensor:s_vec axes:@[ @(rank - 2) ] name:nil];
        const double rtol = (double)std::max(m, n) * 1.1920929e-7;
        MPSGraphTensor* cutoff =
            [g multiplicationWithPrimaryTensor:max_s
                               secondaryTensor:[g constantWithScalar:rtol dataType:s_vec.dataType]
                                          name:nil];
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:s_vec.dataType];
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:s_vec.dataType];
        MPSGraphTensor* keep = [g greaterThanWithPrimaryTensor:s_vec
                                               secondaryTensor:cutoff
                                                          name:nil];
        MPSGraphTensor* safe_s = [g selectWithPredicateTensor:keep
                                          truePredicateTensor:s_vec
                                         falsePredicateTensor:one
                                                         name:nil];
        MPSGraphTensor* inv_s = [g selectWithPredicateTensor:keep
                                         truePredicateTensor:[g divisionWithPrimaryTensor:one
                                                                          secondaryTensor:safe_s
                                                                                     name:nil]
                                        falsePredicateTensor:zero
                                                        name:nil];
        MPSGraphTensor* inv_s_row = [g expandDimsOfTensor:inv_s axes:@[ @(rank - 2) ] name:nil];
        MPSGraphTensor* v_scaled = [g multiplicationWithPrimaryTensor:swap_last_two(g, vh_mat)
                                                      secondaryTensor:inv_s_row
                                                                 name:nil];
        MPSGraphTensor* pinv = [g matrixMultiplicationWithPrimaryTensor:v_scaled
                                                        secondaryTensor:swap_last_two(g, u_mat)
                                                                   name:@"pinv"];
        ctx.bind(node.outputs[0].id, (__bridge void*)from_f32(g, pinv, orig_dt));
        return true;
    }
};

struct DecompositionsRegistrar {
    DecompositionsRegistrar() {
        register_emitter(std::make_unique<SolveTriangularEmitter>());
        register_emitter(std::make_unique<CholeskyEmitter>());
        register_emitter(std::make_unique<QrEmitter>());
        register_emitter(std::make_unique<EighEmitter>());
        register_emitter(std::make_unique<SvdEmitter>());
        register_emitter(std::make_unique<PinvEmitter>());
    }
};

[[maybe_unused]] static const DecompositionsRegistrar g_decompositions_registrar;

}  // namespace

}  // namespace lucid::compile
