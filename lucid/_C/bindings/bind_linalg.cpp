// lucid/_C/bindings/bind_linalg.cpp
//
// Registers dense linear-algebra ops on the `lucid._C.engine.linalg`
// sub-module (created in bind.cpp).  All ops dispatch through the backend
// Dispatcher: CPU uses LAPACK (Accelerate) and GPU uses MLX linalg.
//
// Ops that return multiple tensors (qr → (Q, R), eig/eigh → (values, vectors))
// are wrapped in lambdas that unpack the C++ std::vector result into Python
// tuples, since pybind11 cannot auto-convert a fixed-size vector to tuple.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/linalg/Cholesky.h"
#include "../ops/linalg/Det.h"
#include "../ops/linalg/Eig.h"
#include "../ops/linalg/Eigh.h"
#include "../ops/linalg/HouseholderProduct.h"
#include "../ops/linalg/Inv.h"
#include "../ops/linalg/LDLFactor.h"
#include "../ops/linalg/LUFactor.h"
#include "../ops/linalg/LUSolve.h"
#include "../ops/linalg/Norm.h"
#include "../ops/linalg/Pinv.h"
#include "../ops/linalg/QR.h"
#include "../ops/linalg/SVD.h"
#include "../ops/linalg/Solve.h"
#include "../ops/linalg/SolveTriangular.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers all linalg ops on the `linalg` sub-module.
void register_linalg(py::module_& m) {
    // Single-output ops delegate directly to their C++ op functions.
    m.def("inv", &inv_op, py::arg("a"));
    m.def("det", &det_op, py::arg("a"));
    m.def("solve", &solve_op, py::arg("a"), py::arg("b"));
    m.def("cholesky", &cholesky_op, py::arg("a"), py::arg("upper") = false);

    // norm accepts an axis list; empty list means full-tensor norm.
    // std::move transfers the locally-constructed vector into the C++ op to
    // avoid a redundant copy.
    m.def(
        "norm",
        [](const TensorImplPtr& a, double ord, std::vector<int> dim, bool keepdims) {
            return norm_op(a, ord, std::move(dim), keepdims);
        },
        py::arg("a"), py::arg("ord") = 2.0, py::arg("dim") = std::vector<int>{},
        py::arg("keepdims") = false);

    m.def("p_norm", &p_norm_op, py::arg("a"), py::arg("p"), py::arg("dim"), py::arg("keepdims"));

    // qr returns (Q, R); the C++ op returns a 2-element vector.
    m.def(
        "qr",
        [](const TensorImplPtr& a) {
            auto r = qr_op(a);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"));

    // svd is registered directly; it returns a tuple internally via its own
    // Python-friendly signature.
    m.def("svd", &svd_op, py::arg("a"), py::arg("compute_uv") = true);
    m.def("pinv", &pinv_op, py::arg("a"));

    // eig returns (eigenvalues, eigenvectors) for general (possibly complex)
    // square matrices.
    m.def(
        "eig",
        [](const TensorImplPtr& a) {
            auto r = eig_op(a);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"));

    // eigh is the symmetric/Hermitian specialisation; it guarantees real
    // eigenvalues sorted in ascending order and is faster than eig.
    m.def(
        "eigh",
        [](const TensorImplPtr& a, const std::string& UPLO) {
            auto r = eigh_op(a, UPLO);
            return py::make_tuple(r[0], r[1]);
        },
        py::arg("a"), py::arg("UPLO") = "L",
        "Symmetric/Hermitian eigendecomposition.\n"
        "Returns (eigenvalues, eigenvectors) with eigenvalues sorted ascending.\n"
        "Input must be a real symmetric square matrix.\n"
        "CPU: LAPACK ssyev/dsyev.  GPU: mlx::core::linalg::eigh.");

    // lu_factor: packed LU factorisation + pivots + getrf's info.
    m.def(
        "lu_factor",
        [](const TensorImplPtr& a) {
            auto r = lu_factor_op(a);
            return py::make_tuple(r[0], r[1], r[2]);
        },
        py::arg("a"),
        "LU factorisation with partial pivoting.\n"
        "Returns (LU_packed, pivots, info): LU_packed in LAPACK dgetrf format,\n"
        "pivots int32 1-based, info int32 per matrix (getrf's code; a singular\n"
        "matrix is factored, not refused).");

    // solve_triangular: triangular system solve.
    m.def("solve_triangular", &solve_triangular_op, py::arg("a"), py::arg("b"),
          py::arg("upper") = true, py::arg("unitriangular") = false,
          "Triangular solve: compute X such that A X = B.\n"
          "upper=True  → A is upper triangular.\n"
          "unitriangular=True → A has implicit unit diagonal.");

    // lu_solve: solve AX = B given the packed LU factorisation from lu_factor.
    m.def("lu_solve", &lu_solve_op, py::arg("LU"), py::arg("pivots"), py::arg("b"),
          "Solve AX = B given packed LU (from lu_factor) and pivot vector.\n"
          "CPU: LAPACK sgetrs/dgetrs.  GPU: CPU fallback.");

    // householder_product: recover Q from the compact Householder reflectors.
    m.def("householder_product", &householder_product_op, py::arg("H"), py::arg("tau"),
          "Reconstruct Q from Householder reflectors (H, tau) as returned by\n"
          "LAPACK dgeqrf.  CPU: LAPACK sorgqr/dorgqr.  GPU: CPU fallback.");

    // ldl_factor: LDL^T factorisation for symmetric matrices + sytrf's info.
    m.def(
        "ldl_factor",
        [](const TensorImplPtr& a) {
            auto r = ldl_factor_op(a);
            return py::make_tuple(r[0], r[1], r[2]);
        },
        py::arg("a"),
        "LDL^T factorisation for symmetric matrices.\n"
        "Returns (LD_packed, pivots, info): info is sytrf's int32 code per\n"
        "matrix.  CPU: LAPACK ssytrf/dsytrf.  GPU: CPU fallback.");
}

}  // namespace lucid::bindings
