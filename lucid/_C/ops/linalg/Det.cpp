// lucid/_C/ops/linalg/Det.cpp
//
// Implementation of the determinant op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_det() via Dispatcher.
//   CPU path: LAPACK dgetrf computes an LU factorisation; the determinant is
//             the product of the diagonal elements of U, adjusted for the sign
//             of the permutation.
//   GPU path: mlx::core::linalg::det() on the CPU stream (see _Detail.h).
//
// Backward: ∂L/∂A = g · cof(A), the cofactor matrix (the adjugate's transpose).
//   While every matrix of the batch is safely invertible it is det(A) · A⁻ᵀ,
//   composed from inv_op and elementwise ops, so it differentiates to any
//   order.  When one is singular (|det| < 100 ε) A⁻ᵀ does not exist but the
//   cofactor still does; it is formed from the SVD instead, and recorded
//   through CofactorBackward so a second derivative is still available.
//
// Output shape: the trailing two matrix dimensions are dropped.  A 3-D input
// [B, N, N] produces a 1-D output [B]; a plain [N, N] input produces a scalar
// (shape []).

#include "Det.h"

#include <cstdint>
#include <limits>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/PhantomMode.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Compare.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/bfunc/Sub.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Predicate.h"
#include "../../ops/ufunc/Reductions.h"
#include "../../ops/ufunc/Transpose.h"
#include "../../ops/utils/Layout.h"
#include "../../ops/utils/Select.h"
#include "../../ops/utils/View.h"
#include "../gfunc/Gfunc.h"
#include "Inv.h"
#include "SVD.h"
#include "_Detail.h"

namespace lucid {

const OpSchema DetBackward::schema_v1{"det", 1, AmpPolicy::KeepInput};
const OpSchema CofactorBackward::schema_v1{"det_cofactor", 1, AmpPolicy::KeepInput};

namespace {

// ``t`` of the batch shape, spread over the trailing matrix axes of ``like``.
TensorImplPtr spread_over_matrix(const TensorImplPtr& t, const Shape& like) {
    Shape kept = t->shape();
    kept.push_back(1);
    kept.push_back(1);
    return broadcast_to_op(reshape_op(t, kept), like);
}

// Whether any determinant of the batch is too close to zero for
// det(A) · A⁻ᵀ: below 100 ε the inverse is not trusted (at an exact zero
// pivot it is refused outright), which is the reference's threshold too.
// The answer steers which formula runs, so it is read on the host — the
// data-dependent carve-out.  This adds no new kind of sync: the invertible
// branch already synchronises inside ``inv``, whose LAPACK call runs on the
// host.  ``det`` accepts only F32/F64 (``float_only``), so those two
// epsilons are the only ones that can apply.
bool any_near_singular(const TensorImplPtr& det) {
    // A phantom has no determinant to test, and both formulas have one shape.
    if (PhantomMode::is_enabled())
        return false;
    NoGradGuard ng;
    const double eps = det->dtype() == Dtype::F64 ? std::numeric_limits<double>::epsilon()
                                                  : std::numeric_limits<float>::epsilon();
    auto flag = any_op(less_op(abs_op(det), full_like_op(det, 100.0 * eps)));
    if (flag->device() == Device::GPU) {
        const CpuStorage host = gpu::download_gpu_to_cpu(std::get<GpuStorage>(flag->storage()), {});
        return *reinterpret_cast<const std::uint8_t*>(host.ptr.get()) != 0;
    }
    return *reinterpret_cast<const std::uint8_t*>(
               std::get<CpuStorage>(flag->storage()).ptr.get()) != 0;
}

// A = U diag(S) Vh with alpha = det(U) det(Vh) (±1), shaped (batch, 1, 1).
struct SvdParts {
    TensorImplPtr U, S, Vh, alpha;
};

SvdParts svd_parts(const TensorImplPtr& a) {
    NoGradGuard ng;
    auto f = svd_op(a, /*compute_uv=*/true);
    auto alpha = mul_op(det_op(f[0]), det_op(f[2]));
    Shape kept = alpha->shape();
    kept.push_back(1);
    kept.push_back(1);
    return {f[0], f[1], f[2], reshape_op(alpha, kept)};
}

// M with M[..., i, m] = s_m off the diagonal and 1 on it, so that a
// product over m excludes s_i without dividing by it.
TensorImplPtr excluding_diagonal(const TensorImplPtr& s) {
    const std::int64_t n = s->shape().back();
    auto eye = eye_op(n, n, 0, s->dtype(), s->device());
    auto off = sub_op(ones_like_op(eye), eye);
    return add_op(mul_op(unsqueeze_op(s, -2), off), eye);
}

int last_axis(const TensorImplPtr& t) {
    return static_cast<int>(t->shape().size()) - 1;
}

// cof(A) = alpha · U diag(p) Vh, p_i = prod_{j != i} s_j — the adjugate's
// transpose at any rank (Higham, "What is the adjugate of a matrix?", 2020).
TensorImplPtr cofactor_value(const SvdParts& f) {
    auto M = excluding_diagonal(f.S);
    auto p = prod_op(M, {last_axis(M)}, false);
    return mul_op(f.alpha, matmul_op(mul_op(f.U, unsqueeze_op(p, -2)), f.Vh));
}

// Q[..., i, j] = prod_{m not in {i, j}} s_m off the diagonal, 0 on it — one
// row per pass, so the memory stays O(n²) per matrix.
TensorImplPtr pairwise_excluded_products(const TensorImplPtr& s) {
    const std::int64_t n = s->shape().back();
    const Dtype dt = s->dtype();
    const Device dev = s->device();
    auto M = excluding_diagonal(s);
    Shape q_shape = M->shape();
    auto Q = zeros_op(q_shape, dt, dev);
    for (std::int64_t i = 0; i < n; ++i) {
        auto e_col = eye_op(1, n, i, dt, dev);   // (1, n): 1 at column i
        auto e_row = eye_op(n, 1, -i, dt, dev);  // (n, 1): 1 at row i
        auto not_i = sub_op(ones_like_op(e_col), e_col);
        auto Mi = add_op(mul_op(M, not_i), e_col);
        auto q_i = mul_op(prod_op(Mi, {last_axis(Mi)}, false), not_i);
        Q = add_op(Q, mul_op(e_row, unsqueeze_op(q_i, -2)));
    }
    return Q;
}

// The directional derivative of cof at A = U diag(s) Vh along G.
TensorImplPtr cofactor_directional(const SvdParts& f, const TensorImplPtr& G) {
    const std::int64_t n = f.S->shape().back();
    auto Gp = matmul_op(mT_op(f.U), matmul_op(G, mT_op(f.Vh)));
    auto Q = pairwise_excluded_products(f.S);
    auto diag_g = unsqueeze_op(diagonal_op(Gp, 0, -2, -1), -1);
    auto eye = eye_op(n, n, 0, G->dtype(), G->device());
    auto d_cof_s = sub_op(mul_op(matmul_op(Q, diag_g), eye), mul_op(Q, mT_op(Gp)));
    return mul_op(f.alpha, matmul_op(matmul_op(f.U, d_cof_s), f.Vh));
}

// cof(A) through the SVD, recording [[CofactorBackward]] when grad is on.
TensorImplPtr cofactor_op(const TensorImplPtr& a) {
    auto out = cofactor_value(svd_parts(a));
    kernel::NaryKernel<CofactorBackward, 1>::wire_autograd(std::make_shared<CofactorBackward>(),
                                                           {a}, out, true);
    return out;
}

// ∂det/∂A · g: det(A) A⁻ᵀ g while every matrix of the batch is safely
// invertible (that form differentiates to any order), the SVD cofactor
// otherwise.  ``det`` must be det(a); under grad mode both forms record.
TensorImplPtr det_gradient(const TensorImplPtr& a,
                           const TensorImplPtr& det,
                           const TensorImplPtr& g,
                           const Shape& a_shape) {
    if (any_near_singular(det))
        return mul_op(spread_over_matrix(g, a_shape), cofactor_op(a));
    return mul_op(spread_over_matrix(mul_op(det, g), a_shape), inv_op(mT_op(a)));
}

}  // namespace

// ∂L/∂A = g · cof(A), with cof(A) = det(A) A⁻ᵀ where A is invertible.
std::vector<Storage> DetBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto ddet = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto det_v = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    return {det_gradient(A, det_v, ddet, input_shapes_[0])->storage()};
}

std::vector<TensorImplPtr> DetBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& a = saved_impl_inputs_[0];
    if (!a)
        ErrorBuilder("det").fail("graph-mode backward is missing its saved input");
    return {det_gradient(a, det_op(a), grad_out, input_shapes_[0])};
}

LUCID_REGISTER_OP(DetBackward)

std::vector<Storage> CofactorBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto G = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    return {cofactor_directional(svd_parts(A), G)->storage()};
}

std::vector<TensorImplPtr> CofactorBackward::apply_for_graph(const TensorImplPtr&) {
    ErrorBuilder("det").not_implemented(
        "a third derivative of det at a singular matrix (create_graph=True through "
        "the second derivative) is not implemented");
    return {};
}

LUCID_REGISTER_OP(CofactorBackward)

// Compute det(A).
//
// The output shape is the input shape with the last two matrix dimensions
// removed.  save_inputs=true is passed to wire_autograd because DetBackward
// needs A to call inv_op; without it saved_inputs_[0] would be empty.
TensorImplPtr det_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "det.a").float_only().square_2d();
    OpScopeFull scope{"det", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    // Drop the last two dims: [B, N, N] -> [B],  [N, N] -> [] (scalar).
    Shape out_shape(sh.begin(), sh.end() - 2);

    // det of a 0x0 matrix is 1, not 0: it is the empty product, the same
    // reason an empty sum is 0.  This is the one degenerate result here
    // that is not simply an empty tensor — the output has no matrix axes
    // left to be empty in.
    if (empty_matrix(sh))
        return ones_op(out_shape, a->dtype(), a->device());

    // The reference passes row-major A directly to column-major LAPACK getrf,
    // which factors A^T (det(A^T) = det(A)); transpose the matrix axes so
    // partial pivoting visits the same pivots and singular signed zeros agree.
    std::vector<int> perm(sh.size());
    for (std::size_t i = 0; i < sh.size(); ++i)
        perm[i] = static_cast<int>(i);
    std::swap(perm[sh.size() - 2], perm[sh.size() - 1]);
    auto& disp = backend::Dispatcher::for_device(a->device());
    Storage a_t = disp.permute(a->storage(), sh, perm, a->dtype());
    Storage out_storage = disp.linalg_det(a_t, sh, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
    }
    auto bwd = std::make_shared<DetBackward>();
    // Save det(A) so the backward can use it as the multiplicative factor.
    bwd->keep_output(out->storage());
    // save_inputs=true: DetBackward needs A to recompute inv(A) in the backward.
    kernel::NaryKernel<DetBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
