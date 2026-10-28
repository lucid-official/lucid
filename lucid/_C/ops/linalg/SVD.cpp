// lucid/_C/ops/linalg/SVD.cpp
//
// Implementation of the Singular Value Decomposition forward op.
//
// Forward:
//   CPU path: LAPACK dgesdd performs a divide-and-conquer SVD, which is
//             typically faster than the QR-iteration based dgesvd for large
//             matrices but requires more temporary workspace.
//   GPU path: mlx::core::linalg::svd() on the CPU stream.
//
// Shape pre-computation: ush, ssh, and vsh are all derived before dispatch so
// the backend receives exact target sizes.  The convention is "Vh" (V-hermitian
// conjugate / transpose), not "V", to match NumPy and reference framework naming.
//
// compute_uv flag: passing false to the backend skips materialising U and Vh,
// which can be significantly faster when only the singular values are needed
// (e.g. for computing the condition number or rank estimation).
//
// Autograd: not wired.  A future backward would use the Papadopoulo-Lourakis
// formula; see SVD.h for the derivation sketch.

#include "SVD.h"

#include <algorithm>
#include <variant>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

// Compute the SVD of a.
//
// Shapes (ush, ssh, vsh) are derived before the dispatch call and passed to
// linalg_svd() so both sides agree on memory layout.  The backend returns a
// vector of Storage objects: one element when compute_uv=false, three when
// compute_uv=true (order: [U, S, Vh]).
std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a_in, bool compute_uv) {
    using namespace linalg_detail;
    Validator::input(a_in, "svd.a").non_null();
    if (a_in->shape().size() < 2)
        ErrorBuilder("svd").invalid_argument("input must be at least 2-D");
    const AmpScope amp{"svd", a_in->device()};
    const TensorImplPtr a = amp.cast(a_in);
    require_float(a->dtype(), "svd");
    OpScopeFull scope{"svd", a->device(), a->dtype(), a->shape()};
    scope.set_attr("compute_uv", compute_uv);

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    // k = min(m, n): number of singular values in the reduced decomposition.
    const int k = std::min(m, n);

    // Pre-compute shapes; batch dims are copied from the leading dimensions.
    Shape ush(sh.begin(), sh.end() - 2);
    ush.push_back(m);  // U: m rows
    ush.push_back(k);  // U: k columns (left singular vectors)
    Shape ssh(sh.begin(), sh.end() - 2);
    ssh.push_back(k);  // S: k singular values per matrix
    Shape vsh(sh.begin(), sh.end() - 2);
    vsh.push_back(k);  // Vh: k rows (right singular vectors, transposed)
    vsh.push_back(n);  // Vh: n columns

    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    if (empty_matrix(sh)) {
        std::vector<TensorImplPtr> empty;
        if (compute_uv)
            empty.push_back(zeros_op(ush, a->dtype(), a->device()));
        empty.push_back(zeros_op(ssh, a->dtype(), a->device()));
        if (compute_uv)
            empty.push_back(zeros_op(vsh, a->dtype(), a->device()));
        return empty;
    }

    auto storages = backend::Dispatcher::for_device(a->device())
                        .linalg_svd(a->storage(), sh, compute_uv, ush, ssh, vsh, a->dtype());
    std::vector<TensorImplPtr> out;
    if (!compute_uv) {
        // Values-only mode: backend returns exactly one Storage (singular values).
        out.push_back(fresh(std::move(storages[0]), ssh, a->dtype(), a->device()));
        if (auto* trc = compile::current_tracer())
            trc->on_op_io({a}, out[0]);
        return out;
    }
    // Full SVD mode: backend returns [U storage, S storage, Vh storage].
    out.push_back(fresh(std::move(storages[0]), ush, a->dtype(), a->device()));  // U
    out.push_back(fresh(std::move(storages[1]), ssh, a->dtype(), a->device()));  // S
    out.push_back(fresh(std::move(storages[2]), vsh, a->dtype(), a->device()));  // Vh
    if (auto* trc = compile::current_tracer()) {
        trc->on_op_io({a}, out[0]);
        trc->on_op_io({}, out[1]);
        trc->on_op_io({}, out[2]);
    }
    return out;
}

}  // namespace lucid
